"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize as so

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115  # keep safety margin below 120s

    # Target tensor T[i, j, k] = 1 if multiplying A[i,j] * B[j,k], else 0
    # Shape: (n, m, p)
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    def verify_exact(U, V, W, tol=1e-9):
        # Check if U, V, W are close to half-integers
        # And if the decomposition is exact
        for mat in [U, V, W]:
            # Check half-integer constraint
            # Multiply by 2 and check if close to integer
            diff = mat * 2 - np.round(mat * 2)
            if np.max(np.abs(diff)) > 1e-6:
                return False
        
        # Check decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True

    def round_to_half_integer(mat):
        return np.round(mat * 2) / 2.0

    # Try ranks from low to high
    # Lower bound for 3x3x4 is likely around 20-23
    for target_rank in range(20, 37):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for init_seed in range(5):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(init_seed + target_rank)
            
            # Initialize with small random values
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # Flatten parameters
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss(x):
                U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_x = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_x = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_x[i*m+j, :], V_x[j*p+k, :] * W_x[k*n+i, :])
                            err += (val - T[i, j, k])**2
                return err

            try:
                # Use L-BFGS-B
                res = so.minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success and res.fun < 1e-6:
                    # Extract matrices
                    U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_r = round_to_half_integer(U_opt)
                    V_r = round_to_half_integer(V_opt)
                    W_r = round_to_half_integer(W_opt)
                    
                    if verify_exact(U_r, V_r, W_r):
                        return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
            except Exception:
                pass

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
