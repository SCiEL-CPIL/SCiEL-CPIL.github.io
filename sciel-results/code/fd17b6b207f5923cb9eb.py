"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target: find rank-r decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The tensor T is 1 at positions (i,j,k) for all i in [0,n), j in [0,m), k in [0,p)
    # because each (i,j,k) corresponds to a unique product A[i,j]*B[j,k] contributing to C[i,k]
    
    # For rank r, we need:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For all other (i',j',k') not in the "support": sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] = 0
    
    # Wait, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The tensor is supported on all (i,j,k) triples, and T[i,j,k]=1 for all valid (i,j,k)
    # But the decomposition must satisfy: for each (i,j,k), the sum equals 1
    # And there are no other constraints - the tensor is exactly 1 on all (i,j,k) in [0,n)x[0,m)x[0,p)
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in the domain
    # because each triple (i,j,k) corresponds to exactly one product term.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 24 equations in r*(n*m + m*p + p*n) variables
    
    # Let's try rank 18
    def try_rank(r):
        if time.time() > deadline:
            return None
        
        # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
        # Total variables: r*(n*m + m*p + p*n)
        num_vars = r * (n * m + m * p + p * n)
        
        # Initial guess: schoolbook-like but with some sharing
        # Use random initialization with small values
        np.random.seed(42)
        
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Initialize with a structured guess
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Base: assign each (i,j,k) to a rank term
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
            
            # Add random perturbations to remaining columns
            for c in range(col, r):
                U[:, c] = np.random.randn(n * m) * 0.1
                V[:, c] = np.random.randn(m * p) * 0.1
                W[:, c] = np.random.randn(p * n) * 0.1
            
            def residuals(x):
                U_flat = x[:r * n * m].reshape(n * m, r)
                V_flat = x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
                W_flat = x[r * n * m + r * m * p:].reshape(p * n, r)
                
                res = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            val = np.sum(U_flat[i * m + j] * V_flat[j * p + k] * W_flat[k * n + i])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000, xtol=1e-10, ftol=1e-10)
            except Exception:
                continue
            
            if result.cost < 1e-12:
                U_sol = result.x[:r * n * m].reshape(n * m, r)
                V_sol = result.x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
                W_sol = result.x[r * n * m + r * m * p:].reshape(p * n, r)
                
                # Verify
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_sol[i * m + j] * V_sol[j * p + k] * W_sol[k * n + i])
                            if abs(val - 1.0) > 1e-8:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    # Convert to list of lists
                    return (U_sol.tolist(), V_sol.tolist(), W_sol.tolist())
        
        return None
    
    # Try ranks from 18 upward
    for r in [18, 19, 20, 21, 22, 23, 24]:
        result = try_rank(r)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
