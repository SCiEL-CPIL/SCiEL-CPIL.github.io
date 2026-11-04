"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    
    Uses schoolbook (rank 63) as baseline, then attempts randomized search
    for lower rank solutions by fixing U,V and solving linear systems for W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin from 150s limit
    
    # Schoolbook solution: rank = n * m * p = 3 * 3 * 7 = 63
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Quick randomized search for better rank
    search_deadline = start_time + 50  # Spend at most 50s on search
    for R in range(rank - 1, 55, -1):  # Try R from 62 down to 56
        if time.time() > search_deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > search_deadline - 5:
                break
            
            np.random.seed(trial * 1000 + R)
            U_r = np.random.randint(-1, 2, size=(n * m, R))
            V_r = np.random.randint(-1, 2, size=(m * p, R))
            
            W_r = np.zeros((p * n, R), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build linear system: A @ w = ones(m)
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U_r[i * m + j, r] * V_r[j * p + k, r]
                    
                    try:
                        w_sol = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        W_r[k * n + i, :] = w_sol
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W is close to integer or half-integer
            W_rounded = np.round(W_r * 2) / 2
            if not np.allclose(W_r, W_rounded, atol=1e-4):
                continue
            
            # Verify the solution satisfies all equations
            valid_sol = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_rounded[k * n + i, :])
                        if abs(val - 1.0) > 1e-4:
                            valid_sol = False
                            break
                    if not valid_sol:
                        break
                if not valid_sol:
                    break
            
            if valid_sol:
                best_U, best_V, best_W = U_r, V_r, W_rounded
                best_rank = R
                found = True
                break
        
        if found:
            continue  # Try smaller R
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
