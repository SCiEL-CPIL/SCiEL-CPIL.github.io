"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build list of support indices: (i_idx, j_idx, k_idx) where tensor value is 1
    # i_idx = a*m + b, j_idx = b*p + c, k_idx = c*n + a
    support = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a*m + b
                j_idx = b*p + c
                k_idx = c*n + a
                support.append((i_idx, j_idx, k_idx))
    
    # Try to find lower rank decompositions using iterative refinement
    # Start from schoolbook and try to merge/remove terms
    for r in range(rank_school - 1, 50, -1):
        if time.time() > deadline:
            break
        
        # Use structured random initialization based on support
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with small random perturbations around structured solution
            U_try = np.zeros((nm, r))
            V_try = np.zeros((mp, r))
            W_try = np.zeros((pn, r))
            
            # Assign each support element to a random rank component
            assignments = np.random.randint(0, r, len(support))
            for idx, (i_idx, j_idx, k_idx) in enumerate(support):
                r_idx = assignments[idx]
                U_try[i_idx, r_idx] = 1.0
                V_try[j_idx, r_idx] = 1.0
                W_try[k_idx, r_idx] = 1.0
            
            # Add small random noise to allow optimization
            noise_scale = 0.1
            U_try += noise_scale * np.random.randn(nm, r)
            V_try += noise_scale * np.random.randn(mp, r)
            W_try += noise_scale * np.random.randn(pn, r)
            
            # Iteratively refine using gradient-like updates
            # We want sum_r U[i,r]*V[j,r]*W[k,r] = 1 for support, 0 otherwise
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current reconstruction on support
                errors = []
                for (i_idx, j_idx, k_idx) in support:
                    recon = np.dot(U_try[i_idx, :] * V_try[j_idx, :], W_try[k_idx, :])
                    errors.append(recon - 1.0)
                
                err = np.array(errors)
                if np.max(np.abs(err)) < 1e-10:
                    # Check off-support
                    valid_off = True
                    for _ in range(50):
                        ii = np.random.randint(0, nm)
                        jj = np.random.randint(0, mp)
                        kk = np.random.randint(0, pn)
                        a, b = divmod(ii, m)
                        b2, c = divmod(jj, p)
                        c2, a2 = divmod(kk, n)
                        if b != b2 or c != c2 or a != a2:
                            recon = np.dot(U_try[ii, :] * V_try[jj, :], W_try[kk, :])
                            if abs(recon) > 1e-8:
                                valid_off = False
                                break
                    if valid_off:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = r
                        found = True
                        break
                
                # Update: simple coordinate descent on rank components
                for r_idx in range(r):
                    # Compute contribution of this rank component
                    # For each support element assigned or contributing to r_idx
                    u_col = U_try[:, r_idx]
                    v_col = V_try[:, r_idx]
                    w_col = W_try[:, r_idx]
                    
                    # Gradient-like update: reduce error
                    # This is a simplified approach - just add small random updates
                    U_try[:, r_idx] += lr * np.random.randn(nm) * 0.01
                    V_try[:, r_idx] += lr * np.random.randn(mp) * 0.01
                    W_try[:, r_idx] += lr * np.random.randn(pn) * 0.01
            
            if found:
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
