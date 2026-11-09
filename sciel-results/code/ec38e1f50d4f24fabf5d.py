"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16

    # Build the multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Strassen rank-7 products for 2x2x2
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def build_strassen_solution():
        rank = 56
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)

        def add_strassen_block(a_col0, b_row0, b_col0, col_start):
            c = col_start
            for u_coeff, v_coeff, w_coeff in strassen_products:
                for (li, lj), coeff in u_coeff.items():
                    gi = li
                    gj = a_col0 + lj
                    U[gi * m + gj, c] = coeff
                for (lj, lk), coeff in v_coeff.items():
                    gj = b_row0 + lj
                    gk = b_col0 + lk
                    V[gj * p + gk, c] = coeff
                for (lk, li), coeff in w_coeff.items():
                    gk = b_col0 + lk
                    gi = li
                    W[gk * n + gi, c] = coeff
                c += 1
            return c

        col = 0
        col = add_strassen_block(0, 0, 0, col)
        col = add_strassen_block(2, 2, 0, col)
        col = add_strassen_block(0, 0, 2, col)
        col = add_strassen_block(2, 2, 2, col)
        col = add_strassen_block(0, 0, 4, col)
        col = add_strassen_block(2, 2, 4, col)
        col = add_strassen_block(0, 0, 6, col)
        col = add_strassen_block(2, 2, 6, col)
        return U, V, W

    def verify(U_arr, V_arr, W_arr):
        rank = U_arr.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        return np.array_equal(T_recon, T)

    # Start with Strassen
    U, V, W = build_strassen_solution()
    assert verify(U, V, W), "Strassen solution failed"
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 56

    # Try to reduce rank by attempting lower-rank solutions
    # Use a structured approach: try rank 48, 49, 50, 51, 52, 53, 54, 55
    
    for target_rank in [48, 49, 50, 51, 52, 53, 54, 55]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            
            # Initialize by taking the Strassen solution and randomly zeroing some columns
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Randomly select columns to keep
            cols_to_keep = np.random.choice(56, size=target_rank, replace=False)
            U_try = U_try[:, cols_to_keep]
            V_try = V_try[:, cols_to_keep]
            W_try = W_try[:, cols_to_keep]
            
            # Local search: try to fix the residual by modifying entries
            for it in range(200):
                if time.time() > deadline - 20:
                    break
                
                T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
                for r in range(target_rank):
                    T_recon += np.outer(np.outer(U_try[:, r], V_try[:, r]), W_try[:, r])
                residual = T - T_recon
                
                if np.all(residual == 0):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = target_rank
                    found = True
                    break
                
                # Find worst entry and try to fix
                abs_res = np.abs(residual)
                max_idx = np.unravel_index(np.argmax(abs_res), abs_res.shape)
                a, b, c = max_idx
                target_val = T[a, b, c]
                current_val = T_recon[a, b, c]
                diff = target_val - current_val
                
                if diff == 0:
                    continue
                
                # Try to adjust a random factor
                r = np.random.randint(0, target_rank)
                # Try adjusting U[a, r]
                for delta in [diff, -diff, 1, -1]:
                    old_val = U_try[a, r]
                    U_try[a, r] = old_val + delta
                    # Recompute only affected part
                    T_recon_new = T_recon.copy()
                    T_recon_new += delta * np.outer(np.outer(1, V_try[:, r]), W_try[:, r])
                    # Wait, this is wrong. U[a,r] affects T_recon[a, :, :] only through V and W.
                    # T_recon[a, b2, c2] += delta * V_try[b2, r] * W_try[c2, r]
                    # So:
                    T_recon_new = T_recon + delta * np.outer(V_try[:, r], W_try[:, r])[np.newaxis, :, :]
                    # No, that's also wrong. Let me just recompute.
                    
                    new_err = np.sum((T - (T_recon + delta * np.outer(V_try[:, r], W_try[:, r]) * 0 + delta * np.einsum('br,cr->bcr', V_try[:, r], W_try[:, r])) ** 2)
                    # This is getting messy. Let me just recompute fully.
                    T_recon_full = np.zeros((nm, mp, pn), dtype=np.int64)
                    for r2 in range(target_rank):
                        T_recon_full += np.outer(np.outer(U_try[:, r2], V_try[:, r2]), W_try[:, r2])
                    new_err = np.sum((T - T_recon_full) ** 2)
                    old_err = np.sum((T - T_recon) ** 2)
                    
                    if new_err < old_err:
                        T_recon = T_recon_full
                        break
                    else:
                        U_try[a, r] = old_val
                
                if np.array_equal(verify(U_try, V_try, W_try), True):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
