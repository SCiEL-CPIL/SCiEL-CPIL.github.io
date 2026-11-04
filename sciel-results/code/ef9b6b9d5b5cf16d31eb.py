import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140.0
    
    # Helper to verify a decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)."""
        rank = len(U[0]) if U else 0
        if rank == 0: return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    def make_schoolbook_block(n, m, p, offset_i=0, offset_j=0, offset_k=0):
        """
        Creates a schoolbook decomposition for a sub-block.
        This is for the 3x3x2 part if we split 3x3x5.
        """
        # For 3x3x2, we just use the standard 18 terms.
        # U shape (9, 18), V shape (6, 18), W shape (6, 18) ? No.
        # The full problem is 3x3x5.
        # If we split into 3x3x3 and 3x3x2, we need to map the indices correctly.
        
    # Strategy:
    # 1. Try to find a low-rank decomposition for 3x3x3 (target rank 23-25).
    # 2. If found, append the schoolbook decomposition for 3x3x2 (rank 18).
    # 3. If not found, fallback to full schoolbook 3x3x5 (rank 45).
    
    # We will search for 3x3x3 decomposition.
    # n=3, m=3, p=3.
    # U: 9 x R, V: 9 x R, W: 9 x R.
    
    from scipy.optimize import least_squares
    
    def residuals_333(params, R):
        nm, mp, pn = 3 * 3, 3 * 3, 3 * 3
        U_flat = params[:nm * R].reshape(nm, R)
        V_flat = params[nm * R:nm * R + mp * R].reshape(mp, R)
        W_flat = params[nm * R + mp * R:].reshape(pn, R)
        
        res = np.zeros(27)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 3 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    best_U_333, best_V_333, best_W_333, best_rank_333 = None, None, None, 27
    
    # Try ranks 23 to 26 for 3x3x3
    for target_rank in range(23, 27):
        if time.time() > deadline - 30:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.default_rng(42 + restart)
            U_init = rng.normal(size=(9, target_rank)) * 0.5
            V_init = rng.normal(size=(9, target_rank)) * 0.5
            W_init = rng.normal(size=(9, target_rank)) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals_333(p, target_rank),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=5000
                )
                
                if res.cost < 1e-10:
                    U_sol = res.x[:9 * target_rank].reshape(9, target_rank)
                    V_sol = res.x[9 * target_rank:9 * target_rank + 9 * target_rank].reshape(9, target_rank)
                    W_sol = res.x[9 * target_rank + 9 * target_rank:].reshape(9, target_rank)
                    
                    # Round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, 3, 3, 3):
                        best_U_333, best_V_333, best_W_333, best_rank_333 = U_list, V_list, W_list, target_rank
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            continue

    # If we found a 3x3x3 decomposition, construct the full 3x3x5 solution.
    # Full solution = 3x3x3 part + 3x3x2 part.
    # 3x3x2 part is schoolbook rank 18.
    # We need to combine them.
    
    if best_U_333 is not None:
        # Construct 3x3x2 schoolbook
        # n=3, m=3, p=2.
        # U_2: 9 x 18, V_2: 6 x 18, W_2: 6 x 18.
        # Wait, the dimensions for the full problem are:
        # U: (9, R_total)
        # V: (15, R_total)  # m*p = 3*5
        # W: (15, R_total)  # p*n = 5*3
        
        R_333 = best_rank_333
        R_332 = 18
        R_total = R_333 + R_332
        
        U_full = [[0] * R_total for _ in range(9)]
        V_full = [[0] * R_total for _ in range(15)]
        W_full = [[0] * R_total for _ in range(15)]
        
        # Copy 3x3x3 part (first R_333 columns)
        # Indices for 3x3x3:
        # U[i*3+j, c]
        # V[j*3+k, c]  (k is 0..2)
        # W[k*3+i, c]  (k is 0..2)
        
        for i in range(3):
            for j in range(3):
                for c in range(R_333):
                    U_full[i * 3 + j][c] = best_U_333[i * 3 + j][c]
        
        for j in range(3):
            for k in range(3):
                for c in range(R_333):
                    V_full[j * 5 + k][c] = best_V_333[j * 3 + k][c]
        
        for k in range(3):
            for i in range(3):
                for c in range(R_333):
                    W_full[k * 3 + i][c] = best_W_333[k * 3 + i][c]
        
        # Add 3x3x2 schoolbook part (next R_332 columns)
        # k is 3,4.
        # U[i*3+j, c+R_333] = 1 if term exists?
        # Schoolbook for 3x3x2:
        # For each i, j, k in {3,4}:
        # U[i*3+j, idx] = 1
        # V[j*5+k, idx] = 1
        # W[k*3+i, idx] = 1
        
        idx = R_333
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    U_full[i * 3 + j][idx] = 1
                    V_full[j * 5 + k][idx] = 1
                    W_full[k * 3 + i][idx] = 1
                    idx += 1
        
        # Verify full decomposition
        if verify_decomposition(U_full, V_full, W_full, 3, 3, 5):
            return {"U": U_full, "V": V_full, "W": W_full}
        else:
            # Fallback if verification failed (shouldn't happen)
            pass

    # Fallback to full schoolbook
    U, V, W, rank = make_schoolbook(3, 3, 5)
    return {"U": U, "V": V, "W": W}
