import time
import numpy as np
from scipy.optimize import least_squares
import itertools

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=4, p=4
    # U: 8 x r
    # V: 16 x r
    # W: 8 x r
    
    # Build the target tensor T[i,j,k]
    # i = row_a * 4 + col_a (A is 2x4)
    # j = row_b * 4 + col_b (B is 4x4)
    # k = col_c * 2 + row_c (C is 2x4, trace convention W[k*n+i,r])
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0
    T_flat = T.flatten()

    def compute_residuals(flat_params, U_shape, V_shape, W_shape, T_flat_target):
        # Unpack flat_params into U, V, W
        # Sizes: U: n*m*r, V: m*p*r, W: p*n*r
        r = U_shape[1]
        size_U = U_shape[0] * r
        size_V = V_shape[0] * r
        
        U_flat = flat_params[:size_U]
        V_flat = flat_params[size_U:size_U + size_V]
        W_flat = flat_params[size_U + size_V:]
        
        U = U_flat.reshape(U_shape)
        V = V_flat.reshape(V_shape)
        W = W_flat.reshape(W_shape)
        
        # Compute tensor decomposition
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Vectorized computation:
        # For each r, term_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
        # This is an outer product of U[:,r], V[:,r], W[:,r]
        
        # Efficient way:
        # Reshape U to (8, r), V to (16, r), W to (8, r)
        # We want sum_r U[i,r] V[j,r] W[k,r]
        
        # Let's construct the full tensor via einsum or loops
        # Since r is small (<=28), loops are fine, but let's try vectorization
        # T_calc[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
        
        # Using einsum: 'ir,jr,kr->ijk'
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        return (T_calc.flatten() - T_flat_target)

    def try_rank(r, init_U, init_V, init_W, max_iter=1000):
        """
        Attempt to find a rank-r decomposition starting from init_U, init_V, init_W.
        Returns (success, U, V, W)
        """
        if time.time() > deadline - 5:
            return False, None, None, None
            
        U_shape = (8, r)
        V_shape = (16, r)
        W_shape = (8, r)
        
        # Initialize parameters
        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        # Optimization
        try:
            # Use 'trf' method for box constraints if needed, but default 'lm' is fast for small problems
            # However, 'lm' doesn't support bounds. We don't need bounds strictly, just minimization.
            # If the solution blows up, it's likely not valid.
            result = least_squares(
                compute_residuals, 
                x0, 
                args=(U_shape, V_shape, W_shape, T_flat),
                method='lm',
                max_nfev=2000
            )
        except Exception:
            return False, None, None, None

        if time.time() > deadline - 5:
            return False, None, None, None

        # Check residual
        final_res = np.linalg.norm(result.fun)
        # The tensor has 32 ones. So max error per entry ~ 1e-6 is good.
        # Total residual norm should be very small.
        if final_res < 1e-8:
            # Extract U, V, W
            size_U = U_shape[0] * r
            size_V = V_shape[0] * r
            U_opt = result.x[:size_U].reshape(U_shape)
            V_opt = result.x[size_U:size_U + size_V].reshape(V_shape)
            W_opt = result.x[size_U + size_V:].reshape(W_shape)
            
            # Verify exactly
            T_calc = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
            err = np.max(np.abs(T_calc - T))
            if err < 1e-9:
                return True, U_opt, V_opt, W_opt
            
        return False, None, None, None

    # --- Step 1: Build the rank-28 Strassen decomposition ---
    # Strassen 2x2x2 factors
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)

    # Derive W_s by solving the linear system for 2x2x2
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0

    M_sys = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                T_vec[eq] = T_2[ii, jj, kk]
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

    W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec, rcond=None)
    W_s = W_vec.reshape(4, 7)

    # Verify Strassen decomposition
    err_s = 0.0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                val = np.sum(U_s[ii, :] * V_s[jj, :] * W_s[kk, :])
                err_s = max(err_s, abs(val - T_2[ii, jj, kk]))

    if err_s > 1e-9:
        # Fallback: schoolbook rank 32
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Build the rank-28 Strassen decomposition for 2x4x4
    U28 = np.zeros((8, 28), dtype=float)
    V28 = np.zeros((16, 28), dtype=float)
    W28 = np.zeros((8, 28), dtype=float)

    offset = 0
    for bi in range(2):
        for bj in range(2):
            for ra in range(2):
                for ca in range(2):
                    local_i = ra * 2 + ca
                    global_i = ra * 4 + (2 * bi + ca)
                    U28[global_i, offset:offset + 7] = U_s[local_i, :]
            for rb in range(2):
                for cb in range(2):
                    local_j = rb * 2 + cb
                    global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                    V28[global_j, offset:offset + 7] = V_s[local_j, :]
            for rc in range(2):
                for cc in range(2):
                    local_k = cc * 2 + rc
                    global_k = (2 * bj + cc) * 2 + rc
                    W28[global_k, offset:offset + 7] = W_s[local_k, :]
            offset += 7

    # Verify rank-28 decomposition
    def verify_decomposition(U_full, V_full, W_full, T_target):
        T_calc = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
        return np.max(np.abs(T_calc - T_target))

    err28 = verify_decomposition(U28, V28, W28, T)
    if err28 > 1e-9:
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = 28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()

    # --- Step 2: Non-linear search for lower rank ---
    # Try rank 27, then 26, etc.
    # Strategy: Remove a column from rank 28, and optimize the remaining 27 columns.
    # To increase the chance of success, we can also try removing 2 columns at once for rank 26.
    
    # First, try rank 27 by removing each single column and optimizing.
    found_27 = False
    for c_remove in range(28):
        if time.time() > deadline - 20:
            break
            
        mask = [r for r in range(28) if r != c_remove]
        U27_init = U28[:, mask]
        V27_init = V28[:, mask]
        W27_init = W28[:, mask]
        
        success, U_opt, V_opt, W_opt = try_rank(27, U27_init, V27_init, W27_init)
        
        if success:
            best_rank = 27
            best_U, best_V, best_W = U_opt, V_opt, W_opt
            found_27 = True
            break

    # If rank 27 wasn't found by simple deletion, maybe we need to remove 2 columns for rank 26?
    # Or maybe rank 27 requires a more complex initialization.
    # Let's try a few random perturbations of the rank 27 initialization if we have time.
    if not found_27 and time.time() < deadline - 30:
        # Try a few random subsets of 27 columns? No, that's the same as removing one.
        # Try perturbing the rank 28 solution and then projecting to rank 27?
        # Or just try a few different single-removals with slight noise?
        # Actually, if the true rank is 27, removing the "right" column should work.
        # If it didn't, maybe the true rank is 28.
        
        # Let's try rank 26 by removing 2 columns.
        # There are 28*27/2 = 378 pairs.
        # We can afford to try a subset.
        tried_pairs = 0
        max_pairs_to_try = 50
        
        for c1 in range(28):
            if time.time() > deadline - 10 or tried_pairs >= max_pairs_to_try:
                break
            for c2 in range(c1 + 1, 28):
                if time.time() > deadline - 10 or tried_pairs >= max_pairs_to_try:
                    break
                mask = [r for r in range(28) if r != c1 and r != c2]
                U26_init = U28[:, mask]
                V26_init = V28[:, mask]
                W26_init = W28[:, mask]
                
                success, U_opt, V_opt, W_opt = try_rank(26, U26_init, V26_init, W26_init)
                
                if success:
                    best_rank = 26
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    break
                tried_pairs += 1
            if best_rank == 26:
                break

    # Final verification
    final_err = verify_decomposition(best_U, best_V, best_W, T)
    if final_err > 1e-9:
        # Fallback to schoolbook
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
