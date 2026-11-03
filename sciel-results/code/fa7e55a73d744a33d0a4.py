import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 145.0 # Safety margin

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Start from 54, as Strassen suggests rank can be reduced to 54 for 2x5x6
    # 2x5x6 can be seen as 2x(2+3) x (2+4) or similar.
    # Known lower bound is 32 (Landsberg), upper bound 54 (Strassen extension).
    # Let's try 54 down to 32.
    
    for r in range(54, 31, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Increase restarts for lower ranks as they are harder to find
        num_restarts = 50 if r < 40 else 30
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with random signs, which often works better for 0/1 tensors
            U = np.random.choice([-1, 1], size=(n*m, r)).astype(float)
            V = np.random.choice([-1, 1], size=(m*p, r)).astype(float)
            W = np.random.choice([-1, 1], size=(p*n, r)).astype(float)
            
            err = float('inf')
            converged = False
            
            for it in range(300):
                if time.time() - start_time > time_limit:
                    break
                    
                # 1. Update U
                # A_U[:, c] = flatten(outer(V[:, c], W[:, c]))
                A_U = np.zeros((m*p * p*n, r))
                for c in range(r):
                    A_U[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                
                T_flat = T.reshape(n*m, m*p * p*n)
                try:
                    U_new, residuals, rank, sv = np.linalg.lstsq(A_U, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass

                # 2. Update V
                # A_V[:, c] = flatten(outer(U[:, c], W[:, c]))
                A_V = np.zeros((n*m * p*n, r))
                for c in range(r):
                    A_V[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                
                T_perm_V = np.transpose(T, (1, 0, 2))
                T_flat_V = T_perm_V.reshape(m*p, n*m * p*n)
                
                try:
                    V_new, residuals, rank, sv = np.linalg.lstsq(A_V, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass

                # 3. Update W
                # A_W[:, c] = flatten(outer(U[:, c], V[:, c]))
                A_W = np.zeros((n*m * m*p, r))
                for c in range(r):
                    A_W[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                    
                T_perm_W = np.transpose(T, (2, 0, 1))
                T_flat_W = T_perm_W.reshape(p*n, n*m * m*p)
                
                try:
                    W_new, residuals, rank, sv = np.linalg.lstsq(A_W, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass

                # Check convergence
                T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.linalg.norm(T - T_rec)
                if err < 1e-9:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    break # Found a solution for this rank, try lower
        
    return {"U": best_U, "V": best_V, "W": best_W}
