import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 140  # seconds, with 10s safety margin
    
    def verify_decomposition(U, V, W, n, m, p, rank):
        """Verify that the decomposition is correct."""
        # Random test
        np.random.seed(42)
        A = np.random.randint(-10, 11, (n, m))
        B = np.random.randint(-10, 11, (m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute C from decomposition
        # C[k,i] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] for all j
        # More precisely: C = sum_r (U[:,r] outer V[:,r]) reshaped
        C_computed = np.zeros((p, n))
        for r in range(rank):
            u_col = U_arr[:, r]  # n*m entries
            v_col = V_arr[:, r]  # m*p entries
            w_col = W_arr[:, r]  # p*n entries
            # u_col is a vector in R^{n*m}, v_col in R^{m*p}
            # The product u_col · (A flattened) gives a scalar
            # But we need: for each r, the contribution to C is:
            # C += (u_col reshaped to (n,m)) * (v_col reshaped to (m,p)) * w_col (element-wise broadcast)
            # Actually: C[k,i] += sum_j A[i,j] * V[j*p+k, r] * W[k*n+i, r]
            # Which is: C += sum_r (A * V_r) * W_r where V_r is (m,p) and W_r is (p,n)
            # More precisely:
            # C[k,i] = sum_r sum_j A[i,j] * V[j*p+k, r] * W[k*n+i, r]
            # = sum_r W[k*n+i, r] * sum_j A[i,j] * V[j*p+k, r]
            # = sum_r W[k*n+i, r] * (A @ V_r)[i,k] where V_r is (m,p)
            
            V_r = v_col.reshape(m, p)
            W_r = w_col.reshape(p, n)
            C_computed += (A @ V_r) * W_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True
    
    # Try Winograd F(4) + schoolbook: rank 28
    # If it fails, fall back to rank 65
    
    # First, try to construct the Winograd F(4) decomposition
    # ... [complex construction] ...
    
    # If time is running out, return the best result
    if time.time() - start_time > DEADLINE:
        # Return current best
        pass
    
    # Fallback: current rank-65 solution
    # ... [existing code] ...
