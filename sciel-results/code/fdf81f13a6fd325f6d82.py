import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Lower bound is n*m + m*p + p*n - 2 = 9 + 12 + 12 - 2 = 31
    start_rank = 31
    max_rank = 36
    deadline = time.time() + 110.0
    
    # Pre-compute indices
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We will try to find a decomposition for ranks from start_rank to max_rank
    for R in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Number of equations: n*m*p
        num_eqs = n * m * p
        # Number of variables in W: R * p * n
        num_vars_w = R * p * n
        
        # We will try a few random initializations of U and V
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random U and V
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            
            # Construct the linear system for W
            # A is num_eqs x num_vars_w
            A = np.zeros((num_eqs, num_vars_w))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation for (i, j, k)
                        # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        for r in range(R):
                            w_idx = r * (p * n) + k * n + i
                            A[eq_idx, w_idx] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            # Solve A W = b
            # Use least squares
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check residual
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-9:
                    continue
            
            # Verify the solution
            W = W_flat.reshape(R, p, n)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[r, k, i]
                        if abs(val - 1.0) > 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists for output
                U_out = U.tolist()
                V_out = V.tolist()
                W_out = W.reshape(R, p * n).T.tolist() # W is (R, p, n) -> (R, p*n). Transpose to (p*n, R)
                return {"U": U_out, "V": V_out, "W": W_out}
                
    # Fallback to schoolbook if search fails
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
# EVOLVE-BLOCK-END
