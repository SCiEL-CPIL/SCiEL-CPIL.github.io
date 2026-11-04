import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 110  # safety margin before 150s kill

    # Build the full target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Helper for vectorized ALS step
    def als_step(T, U, V, W, n, m, p, r):
        """
        Performs one ALS iteration.
        T: (n*m, m*p, p*n)
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        """
        # Update U
        # M_VW shape: (m*p*p*n, r)
        # Row index corresponds to (b, c) flattened.
        # M_VW[(b, c), r] = V[b, r] * W[c, r]
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, r)
        # T_flat shape: (n*m, m*p*p*n)
        # T_flat[a, (b, c)] = T[a, b, c]
        T_flat = T.reshape(n*m, m*p*p*n)
        
        # Solve U * M_VW.T = T_flat
        # This is equivalent to M_VW * U.T = T_flat.T
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_VW, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass

        # Update V
        # M_UW shape: (n*m*p*n, r)
        # Row index corresponds to (a, c) flattened.
        # M_UW[(a, c), r] = U[a, r] * W[c, r]
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, r)
        # T_flat_V shape: (m*p, n*m*p*n)
        # T_flat_V[b, (a, c)] = T[a, b, c]
        # T is (nm, mp, pn). We need to reshape to (mp, nm*pn)
        # T.transpose(1, 0, 2).reshape(mp, nm*pn)
        T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
        
        # Solve V * M_UW.T = T_flat_V
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_UW, T_flat_V.T, rcond=None)
            V = V_new.T
        except:
            pass

        # Update W
        # M_UV shape: (n*m*m*p, r)
        # Row index corresponds to (a, b) flattened.
        # M_UV[(a, b), r] = U[a, r] * V[b, r]
        M_UV = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, r)
        # T_flat_W shape: (p*n, n*m*m*p)
        # T_flat_W[c, (a, b)] = T[a, b, c]
        # T is (nm, mp, pn). We need to reshape to (pn, nm*mp)
        # T.transpose(2, 0, 1).reshape(pn, nm*mp)
        T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
        
        # Solve W * M_UV.T = T_flat_W
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_UV, T_flat_W.T, rcond=None)
            W = W_new.T
        except:
            pass

        return U, V, W

    def calculate_error(T, U, V, W):
        # error = sum (T - sum_r U V W)^2
        # Use einsum for speed
        # Reconstructed T: (nm, mp, pn)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_rec)**2)

    # Strategy 1: Block decomposition - 2x3x9 = 3 independent 2x3x3 blocks
    # Search for rank-r decomposition of 2x3x3, then replicate 3x
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0

    # Try ranks for 2x3x3: known lower bound is around 7-8, schoolbook is 18
    for r_block in range(7, 19):
        if time.time() > deadline - 20:
            break
        r_full = r_block * 3
        if r_full >= best_rank:
            break

        # Try multiple random initializations for 2x3x3
        for trial in range(50): # Increased trials due to speedup
            if time.time() > deadline - 20:
                break

            np.random.seed(42 + r_block * 1000 + trial)
            U2 = np.random.randn(n2*m2, r_block)
            V2 = np.random.randn(m2*p2, r_block)
            W2 = np.random.randn(p2*n2, r_block)

            # ALS iterations for 2x3x3
            success = False
            for iteration in range(500): # Increased iterations
                if time.time() > deadline - 20:
                    break

                U2, V2, W2 = als_step(T2, U2, V2, W2, n2, m2, p2, r_block)
                
                # Check convergence
                error = calculate_error(T2, U2, V2, W2)
                
                if error < 1e-10:
                    success = True
                    break
                elif error > 1e4:
                    break

            if success:
                # Build full 2x3x9 decomposition from 3 blocks
                U_full = np.zeros((n*m, r_full))
                V_full = np.zeros((m*p, r_full))
                W_full = np.zeros((p*n, r_full))

                # U is same for all 3 blocks (depends only on i,j)
                U_full[:, 0:r_block] = U2
                U_full[:, r_block:2*r_block] = U2
                U_full[:, 2*r_block:3*r_block] = U2

                # V depends on (j,k): row index = j*9+k
                for j in range(m):
                    for k in range(p):
                        v_row = j * p + k
                        block = k // p2
                        k_local = k % p2
                        v_233_row = j * p2 + k_local
                        col_start = block * r_block
                        V_full[v_row, col_start:col_start+r_block] = V2[v_233_row, :]

                # W depends on (k,i): row index = k*2+i
                for k in range(p):
                    for i in range(n):
                        w_row = k * n + i
                        block = k // p2
                        k_local = k % p2
                        w_233_row = k_local * n + i
                        col_start = block * r_block
                        W_full[w_row, col_start:col_start+r_block] = W2[w_233_row, :]

                # Verify the full decomposition
                full_error = calculate_error(T, U_full, V_full, W_full)

                if full_error < 1e-8:
                    best_rank = r_full
                    best_U, best_V, best_W = U_full, V_full, W_full
                    break

        if best_rank == r_full:
            break

    # Strategy 2: Direct search on full 2x3x9 if block approach didn't find good rank
    if best_rank == rank_sb:
        for r in range(30, rank_sb):
            if time.time() > deadline - 10:
                break
            if r >= best_rank:
                continue

            for trial in range(30): # Increased trials
                if time.time() > deadline - 10:
                    break

                np.random.seed(1000 + r * 1000 + trial)
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)

                success = False
                for iteration in range(300): # Increased iterations
                    if time.time() > deadline - 10:
                        break

                    U, V, W = als_step(T, U, V, W, n, m, p, r)
                    
                    error = calculate_error(T, U, V, W)
                    
                    if error < 1e-10:
                        success = True
                        break
                    elif error > 1e4:
                        break

                if success:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break

            if best_rank == r:
                break

    # Final safety: if time is running out, return best found
    if time.time() > deadline - 5:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
