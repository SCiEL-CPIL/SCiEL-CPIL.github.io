import time
import numpy as np

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin for 150s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _build_target_tensor():
    """Build the (2,2,7) matrix multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j]B[j,k]."""
    n, m, p = 2, 2, 7
    # T is indexed by (i*m+j, j*p+k, k*n+i) in the flattened convention
    # Actually the tensor T[a,b,c] where a = i*m+j (A index), b = j*p+k (B index), c = k*n+i (C index)
    # T[a,b,c] = 1 if a,b,c are consistent with the multiplication, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T):
    """Verify U, V, W decompose the target tensor."""
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-9)

def _polish_decomposition(U, V, W, T, max_iters=200, tol=1e-12):
    """High-precision polish pass to tighten the decomposition."""
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Update U
        for r in range(U.shape[1]):
            VW = np.outer(V[:, r], W[:, r])
            denom = np.sum(VW ** 2)
            if denom < 1e-15:
                continue
            U[:, r] = np.einsum('abc,bc->a', T, VW) / denom
        
        # Update V
        for r in range(V.shape[1]):
            UW = np.outer(U[:, r], W[:, r])
            denom = np.sum(UW ** 2)
            if denom < 1e-15:
                continue
            V[:, r] = np.einsum('abc,ac->b', T, UW) / denom
        
        # Update W
        for r in range(W.shape[1]):
            UV = np.outer(U[:, r], V[:, r])
            denom = np.sum(UV ** 2)
            if denom < 1e-15:
                continue
            W[:, r] = np.einsum('abc,ab->c', T, UV) / denom
        
        # Check convergence with tighter tolerance
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T_rec - T)
        if residual < tol:
            break
    
    return U, V, W

def _search_rank(T, rank, time_limit=10.0):
    """Try to find a rank-`rank` decomposition of T using ALS."""
    n, m, p = 2, 2, 7
    dim_a, dim_b, dim_c = n*m, m*p, p*n  # 4, 14, 14
    
    start = time.time()
    
    # Try multiple random initializations
    num_trials = 15
    max_iters = 150
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small values
        # Scale to avoid numerical issues
        U = np.random.randn(dim_a, rank) * 0.5
        V = np.random.randn(dim_b, rank) * 0.5
        W = np.random.randn(dim_c, rank) * 0.5
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for r in range(rank):
                VW = np.outer(V[:, r], W[:, r])
                denom = np.sum(VW ** 2)
                if denom < 1e-12:
                    continue
                U[:, r] = np.einsum('abc,bc->a', T, VW) / denom
            
            # Update V
            for r in range(rank):
                UW = np.outer(U[:, r], W[:, r])
                denom = np.sum(UW ** 2)
                if denom < 1e-12:
                    continue
                V[:, r] = np.einsum('abc,ac->b', T, UW) / denom
            
            # Update W
            for r in range(rank):
                UV = np.outer(U[:, r], V[:, r])
                denom = np.sum(UV ** 2)
                if denom < 1e-12:
                    continue
                W[:, r] = np.einsum('abc,ab->c', T, UV) / denom
            
            # Check convergence
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_rec - T)
            if residual < 1e-9:
                # Polish for higher precision
                U, V, W = _polish_decomposition(U, V, W, T)
                if _verify_decomposition(U, V, W, T):
                    return U, V, W
    
    return None

def _construct_rank25():
    """Fallback rank-25 Strassen block decomposition."""
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        # M1 = (a+d)(e+h)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[k_start, col_start+0] += 1
        V[p + k_start + 1, col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (c+d)e
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = a(f-h)
        U[0, col_start+2] += 1
        V[k_start+1, col_start+2] += 1
        V[p + k_start + 1, col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = d(g-e)
        U[3, col_start+3] += 1
        V[p + k_start, col_start+3] += 1
        V[k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (a+b)h
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[p + k_start + 1, col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (c-a)(e+f)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[k_start, col_start+5] += 1
        V[k_start+1, col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (b-d)(g+h)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[p + k_start, col_start+6] += 1
        V[p + k_start + 1, col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    # Schoolbook for k=6
    # C06 = a*B06 + b*B16
    # C16 = c*B06 + d*B16
    
    # Term 1: a * B06
    U[0, col_start+0] += 1
    V[0*p + 6, col_start+0] += 1
    W[6*n + 0, col_start+0] += 1
    
    # Term 2: b * B16
    U[1, col_start+1] += 1
    V[1*p + 6, col_start+1] += 1
    W[6*n + 0, col_start+1] += 1
    
    # Term 3: c * B06
    U[2, col_start+2] += 1
    V[0*p + 6, col_start+2] += 1
    W[6*n + 1, col_start+2] += 1
    
    # Term 4: d * B16
    U[3, col_start+3] += 1
    V[1*p + 6, col_start+3] += 1
    W[6*n + 1, col_start+3] += 1
    
    return U, V, W

def construct():
    T = _build_target_tensor()
    
    # Try to find a lower rank decomposition
    # We prioritize lower ranks
    target_ranks = [22, 23, 24]
    
    for r in target_ranks:
        if _time_remaining() < 10.0:
            break
        # Allocate time based on rank difficulty (lower rank is harder)
        time_alloc = min(15.0, _time_remaining() - 10.0)
        result = _search_rank(T, r, time_limit=time_alloc)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_decomposition(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Rank 25
    U, V, W = _construct_rank25()
    # Verify fallback (should always pass)
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If even fallback fails (should not happen), return zeros (invalid)
    return {"U": [[0]*25 for _ in range(4)], "V": [[0]*25 for _ in range(14)], "W": [[0]*25 for _ in range(14)]}
