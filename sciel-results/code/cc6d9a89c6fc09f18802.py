"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Deadline guard: 150s limit, keep safety margin
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Schoolbook decomposition (rank 45)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify a decomposition is valid
    def verify(U, V, W, max_checks=50):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Check dimensions
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check entries are integer or half-integer
        for matrix in [U, V, W]:
            for val in matrix.flat:
                if val != 0 and abs(val * 2 - round(val * 2)) > 1e-10:
                    return False
        
        # Verify on random matrices
        np.random.seed(42)
        for _ in range(max_checks):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute using factors
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                
                # u_col has n*m entries, reshape to (n, m)
                u_mat = u_col.reshape(n, m)
                # v_col has m*p entries, reshape to (m, p)
                v_mat = v_col.reshape(m, p)
                # w_col has p*n entries, reshape to (p, n)
                w_mat = w_col.reshape(p, n)
                
                # Term for this r: W * (U^T A) * (B V)
                # Actually: C[i,k] += W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
                
                # Let's compute it directly
                ua = np.zeros(n)
                for i in range(n):
                    ua[i] = np.dot(u_mat[i, :], A[i, :])
                
                vb = np.zeros(p)
                for k in range(p):
                    vb[k] = np.dot(v_mat[:, k], B[:, k])
                
                # Now C[i,k] += W[k*n+i, r] * ua[i] * vb[k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += w_mat[k, i] * ua[i] * vb[k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    
    # Try to find a better decomposition
    # Strategy: Start from schoolbook and try to merge terms
    
    # First, let's try a simple optimization: look for rank-1 terms that can be combined
    # This is a heuristic approach
    
    # For now, return schoolbook as it's guaranteed to work
    # Future work: implement border rank perturbation
    
    result = schoolbook()
    
    return result
# EVOLVE-BLOCK-END
