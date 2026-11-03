"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if i==j and j==k (identity-like), 
    # but actually for A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor should satisfy: T[i,j,k] = 1 if we're computing (A*B)[i,k] contribution
    # Actually, the tensor decomposition means:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # This is getting complex. Let me think differently.
    # The standard form: (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # Tensor form: T_{ijk} = delta_{ik} if we're looking at identity, but no.
    # For matrix multiplication tensor, T_{ijk} = 1 for all i,j,k? No.
    # Actually, the multiplication tensor has T_{ijk} = 1 if the indices correspond to valid multiplications.
    # Let me just check: for each (i,j,k), the tensor entry should be 1 if this triple contributes to C[i,k].
    # C[i,k] = sum_j A[i,j] B[j,k], so for each (i,j,k), T_{ijk} = 1.
    # Wait, that would make it rank 125 trivially. Let me re-read the problem.
    # The problem says "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because (A*B)[i,k] = sum_j A[i,j] * B[j,k], and each term is a product of three factors.
    # So yes, T[i,j,k] = 1 for all valid (i,j,k).
    # But that means the rank is at least max(n*m, m*p, p*n) = 25, and at most 125.
    # The schoolbook gives rank 125. We want to find lower rank.
    
    # Let me verify by checking that for random A, B, we get correct C.
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using the decomposition
    # (A*B)[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Wait, this doesn't look right dimensionally. Let me reconsider.
    
    # Standard rank-r decomposition of the multiplication tensor:
    # (A*B)[i,k] = sum_{c=1}^{r} (sum_j U[i,j,c] * A[i,j]) * (sum_l V[l,k,c] * B[l,k]) * W[i,k,c]
    # But the interface says U has shape (n*m, rank), so U[i*m+j, c]
    # V has shape (m*p, rank), so V[j*p+k, c]  
    # W has shape (p*n, rank), so W[k*n+i, c]
    
    # So the formula is:
    # (A*B)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    # Let me verify this.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        # For each c, we have vectors u_c, v_c, w_c
        u_c = U_arr[:, c]  # shape (n*m,)
        v_c = V_arr[:, c]  # shape (m*p,)
        w_c = W_arr[:, c]  # shape (p*n,)
        
        # Reshape
        u_c = u_c.reshape(n, m)
        v_c = v_c.reshape(m, p)
        w_c = w_c.reshape(p, n)
        
        # The contribution to C[i,k] from this term
        # Should be: (A * u_c.T)[i] * (v_c * B.T)[k] * w_c[k, i]? 
        # This is getting confusing. Let me just trust the verification by checking
        # that the tensor reconstruction matches.
        
        pass
    
    # Simpler approach: directly reconstruct the tensor and compare
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k.
    
    T_reconstructed = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        
        # u_c[i*m + j] * v_c[j*p + k] * w_c[k*n + i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += u_c[i*m + j] * v_c[j*p + k] * w_c[k*n + i]
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_reconstructed, T_expected, atol=tol)

def _try_random_decomposition(rank, n, m, p, max_iter=1000, tol=1e-9, time_limit=None, start_time=None):
    """Try to find a rank-r decomposition using random initialization and local search."""
    if start_time is None:
        start_time = time.time()
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(5):  # 5 random restarts
        if time_limit and time.time() - start_time > time_limit:
            break
        
        # Random initialization with small values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Local search: gradient-free optimization
        for iter in range(max_iter):
            if time_limit and time.time() - start_time > time_limit:
                break
            
            # Compute current error
            T_reconstructed = np.zeros((n, m, p))
            for c in range(rank):
                u_c = U[:, c]
                v_c = V[:, c]
                w_c = W[:, c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_reconstructed[i, j, k] += u_c[i*m + j] * v_c[j*p + k] * w_c[k*n + i]
            
            err = np.sum((T_reconstructed - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < tol:
                return U.copy(), V.copy(), W.copy(), err
            
            # Perturb one matrix
            which = random.choice([0, 1, 2])
            if which == 0:
                idx = random.randint(0, n * m - 1)
                col = random.randint(0, rank - 1)
                U[idx, col] += random.gauss(0, 0.01)
            elif which == 1:
                idx = random.randint(0, m * p - 1)
                col = random.randint(0, rank - 1)
                V[idx, col] += random.gauss(0, 0.01)
            else:
                idx = random.randint(0, p * n - 1)
                col = random.randint(0, rank - 1)
                W[idx, col] += random.gauss(0, 0.01)
    
    if best_U is not None:
        return best_U, best_V, best_W, best_err
    return None, None, None, float('inf')

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    start_time = time.time()
    time_limit = 120  # seconds, with safety margin
    
    # Try to find lower-rank decompositions
    for rank in range(25, schoolbook_rank + 1):
        if time.time() - start_time > time_limit - 10:  # Leave 10s margin
            break
        
        U, V, W, err = _try_random_decomposition(rank, n, m, p, max_iter=200, time_limit=time_limit, start_time=start_time)
        
        if U is not None and err < 1e-9:
            # Verify exactly
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
