<<<<<<< SEARCH
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
=======
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
_DEADLINE = 145.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the 3D multiplication tensor T[i, j, k] where T[i,j,k] = 1 if the 
    (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    Actually, the standard matmul tensor has T[i,j,k] = 1 for all valid (i,j,k) 
    since C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor is indexed such that T[i,j,k] = 1 means the term A[i,j]*B[j,k] 
    contributes to C[i,k].
    """
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrix(T, mode):
    """Reshape tensor to matrix for SVD. mode=0: (n, m*p), mode=1: (m, n*p), mode=2: (p, n*m)"""
    n, m, p = T.shape
    if mode == 0:
        return T.reshape(n, m * p)
    elif mode == 1:
        return T.transpose(1, 0, 2).reshape(m, n * p)
    else:
        return T.transpose(2, 0, 1).reshape(p, n * m)

def _rank_estimate(T):
    """Estimate tensor rank via max of matrix mode ranks."""
    r0 = np.linalg.matrix_rank(_tensor_to_matrix(T, 0))
    r1 = np.linalg.matrix_rank(_tensor_to_matrix(T, 1))
    r2 = np.linalg.matrix_rank(_tensor_to_matrix(T, 2))
    return max(r0, r1, r2)

def _peel_rank1(T, n, m, p):
    """Find the best rank-1 approximation to T and return it along with the residual.
    Uses the approach: find u, v, w that maximize |<T, u⊗v⊗w>| / (||u||*||v||*||w||).
    For simplicity, we try the schoolbook basis vectors and also use power iteration.
    """
    best_rank1 = None
    best_score = 0.0
    
    # Try all schoolbook basis vectors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R = np.zeros_like(T)
                R[i, j, k] = 1.0
                score = np.sum(T * R)
                if score > best_score:
                    best_score = score
                    best_rank1 = R
    
    # Also try power iteration for a potentially better rank-1 component
    if best_score > 0:
        u = np.ones(n) / np.sqrt(n)
        v = np.ones(m) / np.sqrt(m)
        w = np.ones(p) / np.sqrt(p)
        
        for _ in range(20):
            # Update u
            uv = np.tensordot(T, v, axes=(1, 0))  # (n, p)
            u_new = np.tensordot(uv, w, axes=(1, 0))  # (n,)
            if np.linalg.norm(u_new) > 1e-10:
                u = u_new / np.linalg.norm(u_new)
            
            # Update v
            uw = np.tensordot(T, u, axes=(0, 0))  # (m, p)
            v_new = np.tensordot(uw, w, axes=(1, 0))  # (m,)
            if np.linalg.norm(v_new) > 1e-10:
                v = v_new / np.linalg.norm(v_new)
            
            # Update w
            uv = np.tensordot(T, u, axes=(0, 0))  # (m, p)
            w_new = np.tensordot(uv, v, axes=(0, 0))  # (p,)
            if np.linalg.norm(w_new) > 1e-10:
                w = w_new / np.linalg.norm(w_new)
        
        R_power = np.outer(np.outer(u, v), w).reshape(n, m, p)
        score_power = np.sum(T * R_power) / (np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(w))
        if score_power > best_score:
            best_rank1 = R_power
            best_score = score_power
    
    residual = T - best_rank1
    return best_rank1, residual

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents the matmul tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T += np.outer(np.outer(u, v), w).reshape(n, m, p)
    
    # Check against the true matmul tensor (all ones for valid indices)
    T_true = np.ones((n, m, p), dtype=np.float64)
    return np.max(np.abs(T - T_true)) < tol

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_peeling(n, m, p, max_rank, start_time):
    """Try to find a decomposition with rank <= max_rank using peeling."""
    T = _build_tensor(n, m, p)
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(max_rank):
        if time.time() - start_time > _DEADLINE - 5:
            return None
        
        rank1, residual = _peel_rank1(T, n, m, p)
        if rank1 is None:
            break
        
        # Extract u, v, w from rank1
        u = rank1.reshape(n, m * p)[:, 0]  # This is wrong, need to extract properly
        # Actually, for schoolbook basis, u[i]=1, v[j]=1, w[k]=1 for specific i,j,k
        # For power iteration, we need to store u, v, w separately
        
        # Let's redo: return u, v, w, residual
        break
    
    return None

def _try_peeling_v2(n, m, p, max_rank, start_time):
    """Improved peeling that tracks u, v, w vectors."""
    T = _build_tensor(n, m, p)
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(max_rank):
        if time.time() - start_time > _DEADLINE - 5:
            return None
        
        # Find best rank-1 component
        best_u, best_v, best_w = None, None, None
        best_score = 0.0
        
        # Try schoolbook basis
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n)
                    v = np.zeros(m)
                    w = np.zeros(p)
                    u[i] = 1
                    v[j] = 1
                    w[k] = 1
                    R = np.outer(np.outer(u, v), w).reshape(n, m, p)
                    score = np.sum(T * R)
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u, v, w
        
        # Try power iteration
        u = np.ones(n) / np.sqrt(n)
        v = np.ones(m) / np.sqrt(m)
        w = np.ones(p) / np.sqrt(p)
        
        for _ in range(30):
            uv = np.tensordot(T, v, axes=(1, 0))
            u_new = np.tensordot(uv, w, axes=(1, 0))
            if np.linalg.norm(u_new) > 1e-10:
                u = u_new / np.linalg.norm(u_new)
            
            uw = np.tensordot(T, u, axes=(0, 0))
            v_new = np.tensordot(uw, w, axes=(1, 0))
            if np.linalg.norm(v_new) > 1e-10:
                v = v_new / np.linalg.norm(v_new)
            
            uv = np.tensordot(T, u, axes=(0, 0))
            w_new = np.tensordot(uv, v, axes=(0, 0))
            if np.linalg.norm(w_new) > 1e-10:
                w = w_new / np.linalg.norm(w_new)
        
        R_power = np.outer(np.outer(u, v), w).reshape(n, m, p)
        score_power = np.sum(T * R_power)
        if score_power > best_score:
            best_score = score_power
            best_u, best_v, best_w = u, v, w
        
        if best_score < 1e-10:
            break
        
        U_list.append(best_u.tolist())
        V_list.append(best_v.tolist())
        W_list.append(best_w.tolist())
        
        # Subtract from T
        R = np.outer(np.outer(best_u, best_v), best_w).reshape(n, m, p)
        T = T - R
        
        # Check if T is approximately zero
        if np.max(np.abs(T)) < 1e-9:
            break
    
    if np.max(np.abs(T)) > 1e-9:
        return None
    
    # Build final matrices
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u = U_list[r]
        v = V_list[r]
        w = W_list[r]
        for i in range(n):
            U[i * m + 0][r] = u[i]  # Wait, this is wrong indexing
        # Actually U should be (n*m, rank) where row index is i*m+j
        # But our u is just (n,), so we need to map it
        # The issue: u is over i only, but U rows are indexed by (i,j)
        # For the standard matmul tensor, the rank-1 component has u[i], v[j], w[k]
        # So U[i*m+j] = u[i] * v[j]? No, that's not right either.
        
        # Let me reconsider: the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For a rank-1 term with vectors u (size n), v (size m), w (size p):
        # U[i*m+j, r] = u[i] * v[j] is NOT correct.
        # Actually, the rank-1 decomposition of the tensor T[i,j,k] = u[i]*v[j]*w[k]
        # means: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = u[i]*v[j]*w[k]
        # This can be achieved by: U[i*m+j, r] = u[i], V[j*p+k, r] = v[j], W[k*n+i, r] = w[k]
        # But then the product is u[i]*v[j]*w[k], which is correct!
        
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = u[i]
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = v[j]
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = w[k]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook solution as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try peeling for lower rank
    # The minimum possible rank is at least max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But for matmul, the rank is typically n*m*p for small dimensions
    # Let's try ranks from 18 up to n*m*p-1
    
    min_possible = max(n * m, m * p, p * n)
    for rank in range(min_possible, n * m * p):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        result = _try_peeling_v2(n, m, p, rank, start_time)
        if result is not None:
            # Verify
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    return schoolbook
>>>>>>> REPLACE

I realize there's an issue with my peeling approach - the way I'm building U, V, W from the u, v, w vectors is incorrect for the tensor decomposition format. Let me fix this properly.

The matmul tensor T[i,j,k] decomposes as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]. For a rank-1 term u[i]*v[j]*w[k], we need U[i*m+j, r] = u[i]*v[j], V[j*p+k, r] = 1, W[k*n+i, r] = w[k] OR other factorizations. Actually, the cleanest is: U[i*m+j, r] = u[i], V[j*p+k, r] = v[j], W[k*n+i, r] = w[k] gives product u[i]*v[j]*w[k] which matches T[i,j,k] for that term.

But wait - that means each rank-1 term contributes u[i]*v[j]*w[k] to T[i,j,k], and the sum over r gives the full tensor. So my construction is actually correct! Let me verify and clean up the code.

