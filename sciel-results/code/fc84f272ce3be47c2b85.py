"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor
    # T[i, j, k] = 1 if i=(a,b), j=(b,c), k=(c,a) for consistent a,b,c
    # else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using random search
    start_time = time.time()
    deadline = start_time + 100
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try ranks from 85 down to 50
    for r in range(85, 49, -5):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U_try = np.random.randn(nm, r)
            V_try = np.random.randn(mp, r)
            W_try = np.random.randn(pn, r)
            
            # Check if this is a valid decomposition
            # Compute the reconstructed tensor
            # T_recon[i,j,k] = sum_r U_try[i,r] * V_try[j,r] * W_try[k,r]
            # This is expensive, so let's check a subset of constraints
            
            # Check the support constraints
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_i = i*m + j
                        idx_j = j*p + k
                        idx_k = k*n + i
                        recon = np.sum(U_try[idx_i, :] * V_try[idx_j, :] * W_try[idx_k, :])
                        if abs(recon - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check a few off-support constraints
            for _ in range(100):
                i = np.random.randint(0, nm)
                j = np.random.randint(0, mp)
                k = np.random.randint(0, pn)
                # Check if this is in the support
                a, b = divmod(i, m)
                b2, c = divmod(j, p)
                c2, a2 = divmod(k, n)
                if b != b2 or c != c2 or a != a2:
                    recon = np.sum(U_try[i, :] * V_try[j, :] * W_try[k, :])
                    if abs(recon) > 1e-6:
                        valid = False
                        break
            if not valid:
                continue
            
            # Found a valid decomposition!
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = r
            break
        if best_rank == r:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
