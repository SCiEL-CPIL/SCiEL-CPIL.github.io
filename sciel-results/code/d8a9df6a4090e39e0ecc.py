"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i, j, k]
    # For standard matrix multiplication, the component T[i, j, k] is 1 for all valid i, j, k
    # because A[i, j] * B[j, k] always contributes to C[i, k].
    T = np.ones((n, m, p), dtype=float)
    
    # Fallback: Schoolbook decomposition (Rank = n * m * p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = schoolbook()
    best_result = (U_s, V_s, W_s, rank_s)
    
    # Function to verify a decomposition
    def verify(U, V, W, r):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - T[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    # Try to find lower rank decompositions
    # Strategy: For a fixed rank r, randomly initialize U and V.
    # Solve for W using least squares.
    # The equation is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # For fixed i, k, this is a linear system in W[k*n+i, :]
    # Let w_ki = W[k*n+i, :]
    # sum_c (U[i*m+j, c] * V[j*p+k, c]) * w_ki[c] = 1 for all j
    # This is A_ki @ w_ki = ones(m)
    # where A_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
    
    # We can try ranks from 1 upwards
    # If r=1 works, we return immediately.
    
    for r in range(1, min(60, n*m*p + 1)):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Number of random restarts
        # For small r, we need many restarts. For larger r, fewer.
        max_restarts = 50 if r < 10 else 5
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V
            # Use small random values to avoid numerical instability
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            
            # Solve for W
            W = np.zeros((pn, r))
            success = True
            
            for k in range(p):
                for i in range(n):
                    # Construct A_ki of shape (m, r)
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        A_ki[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    
                    # Solve A_ki @ w = ones(m)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n + i, :] = w
                    except np.linalg.LinAlgError:
                        success = False
                        break
                if not success:
                    break
            
            if not success:
                continue
                
            # Verify
            err = verify(U, V, W, r)
            if err < 1e-9:
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break

    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
