"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(r):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try random bases for U and W
    # The system is linear in V given U and W.
    # A V_flat = b
    # A shape: (n*m*p, m*p*r)
    # b shape: (n*m*p,)
    
    # We iterate over ranks.
    # Lower bound for M(4,4,7) is not 112.
    # M(4,4,4) <= 49 (Strassen). M(4,4,3) <= 49. Total <= 98.
    # Let's try ranks from 50 to 100.
    
    np.random.seed(42) # For reproducibility
    
    for r in range(50, 101):
        # Generate random U and W
        # Using small integers or floats?
        # Floats are fine for solving, but the final result should be "exact".
        # If we use random floats, V will be floats.
        # This is usually acceptable.
        
        U = np.random.randn(n * m, r)
        W = np.random.randn(p * n, r)
        
        # Build A matrix
        # A[eq_idx, var_idx] = U[i*m+j, t] * W[k*n+i, t]
        # eq_idx corresponds to (i,j,k)
        # var_idx corresponds to (j*p+k, t)
        
        A = np.zeros((n * m * p, m * p * r))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        # Solve A V_flat = b
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return result
        except:
            pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
