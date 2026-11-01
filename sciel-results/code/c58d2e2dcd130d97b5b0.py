import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Build the target tensor T[i,j,k] for 2x4x4
    # i = row_a * 4 + col_a (A is 2x4)
    # j = row_b * 4 + col_b (B is 4x4)
    # k = col_c * 2 + row_c (C is 2x4, stored as W[k*n+i,r] convention: k = col*n + row)
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
    def build_rank28():
        U_full = np.zeros((8, 28), dtype=float)
        V_full = np.zeros((16, 28), dtype=float)
        W_full = np.zeros((8, 28), dtype=float)

        offset = 0
        for bi in range(2):
            for bj in range(2):
                # Map local indices to global
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U_full[global_i, offset:offset + 7] = U_s[local_i, :]

                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V_full[global_j, offset:offset + 7] = V_s[local_j, :]

                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W_full[global_k, offset:offset + 7] = W_s[local_k, :]

                offset += 7

        return U_full, V_full, W_full

    U28, V28, W28 = build_rank28()

    # Verify rank-28 decomposition
    def verify_decomposition(U_full, V_full, W_full, T_target):
        max_err = 0.0
        # Vectorized check for speed
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # We can compute the tensor product and compare
        # But simple loops are fine for 8x16x8 = 1024 elements
        for i in range(U_full.shape[0]):
            for j in range(V_full.shape[0]):
                for k in range(W_full.shape[0]):
                    val = np.dot(U_full[i, :] * V_full[j, :], W_full[k, :])
                    err = abs(val - T_target[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    err28 = verify_decomposition(U28, V28, W28, T)
    
    if err28 > 1e-9:
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

    # NEW STRATEGY: Iterative Rank Reduction (ALS with Rounding)
    # We start with the rank-28 solution, drop one column (term) to get rank 27.
    # Then we iterate:
    # 1. Fix V and W, solve for U.
    # 2. Round U to nearest half-integer.
    # 3. Fix U and W, solve for V.
    # 4. Round V to nearest half-integer.
    # 5. Fix U and V, solve for W.
    # 6. Round W to nearest half-integer.
    # Check if the rounded U, V, W produce the target tensor exactly.
    
    def try_rank_reduction(rank_target=27, num_restarts=50, max_iters=100):
        best_err = float('inf')
        best_solution = None
        
        # We will try dropping different columns from the Strassen decomposition
        # and also try random perturbations of the dropped column's position
        
        for restart in range(num_restarts):
            if time.time() - start_time > deadline:
                break
                
            # Initialize with Strassen, drop one column
            # To diversify, we drop a random column, or we shift the "gap"
            drop_idx = restart % 28
            
            U = U28[:, :drop_idx].copy()
            if drop_idx < 27:
                U = np.hstack([U, U28[:, drop_idx+1:]])
            
            V = V28[:, :drop_idx].copy()
            if drop_idx < 27:
                V = np.hstack([V, V28[:, drop_idx+1:]])
            
            W = W28[:, :drop_idx].copy()
            if drop_idx < 27:
                W = np.hstack([W, W28[:, drop_idx+1:]])
                
            # Add a small random perturbation to avoid local minima/stagnation
            # But keep it small to stay close to the known solution
            if restart > 0:
                U += np.random.normal(0, 0.1, U.shape)
                V += np.random.normal(0, 0.1, V.shape)
                W += np.random.normal(0, 0.1, W.shape)

            for it in range(max_iters):
                if time.time() - start_time > deadline:
                    break
                    
                # 1. Solve for U
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # For fixed j, k: T[:,j,k] = U @ (V[j,:] * W[k,:])^T
                # This is a linear system for U.
                # Let M_U be the matrix where M_U[i, (j,k,r)] = V[j,r] W[k,r]
                # Then T_flat = M_U @ U_flat
                # We can solve this using least squares.
                
                # Build M_U
                # Dimensions: U is 8x27. T is 8x16x8.
                # For each (j,k), we have 8 equations for 27 unknowns? No.
                # U is 8x27. The equations are coupled across j,k.
                # T[i,j,k] = sum_r U[i,r] (V[j,r] W[k,r])
                # Let's vectorize.
                # vec(T) = (W^T \otimes V^T) vec(U^T)? No.
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # This is a linear system in U if V and W are fixed.
                # Let's construct the big matrix A_U such that vec(U) = A_U^+ vec(T)
                # A_U has size (8*16*8) x (8*27)
                # Row index: idx = ((i*16)+j)*8 + k
                # Col index: idx_u = i*27 + r
                # A_U[idx, idx_u] = V[j,r] * W[k,r]
                
                A_U = np.zeros((8*16*8, 8*27), dtype=float)
                T_flat = T.flatten()
                
                for i in range(8):
                    for j in range(16):
                        for k in range(8):
                            row_idx = (i*16 + j)*8 + k
                            for r in range(27):
                                col_idx = i*27 + r
                                A_U[row_idx, col_idx] = V[j,r] * W[k,r]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
                U = U_flat.reshape(8, 27)
                U = np.round(U * 2) / 2  # Round to half-integers
                
                # 2. Solve for V
                # A_V has size (8*16*8) x (16*27)
                # Row index: idx = ((i*16)+j)*8 + k
                # Col index: idx_v = j*27 + r
                # A_V[idx, idx_v] = U[i,r] * W[k,r]
                
                A_V = np.zeros((8*16*8, 16*27), dtype=float)
                for i in range(8):
                    for j in range(16):
                        for k in range(8):
                            row_idx = (i*16 + j)*8 + k
                            for r in range(27):
                                col_idx = j*27 + r
                                A_V[row_idx, col_idx] = U[i,r] * W[k,r]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
                V = V_flat.reshape(16, 27)
                V = np.round(V * 2) / 2
                
                # 3. Solve for W
                # A_W has size (8*16*8) x (8*27)
                # Row index: idx = ((i*16)+j)*8 + k
                # Col index: idx_w = k*27 + r
                # A_W[idx, idx_w] = U[i,r] * V[j,r]
                
                A_W = np.zeros((8*16*8, 8*27), dtype=float)
                for i in range(8):
                    for j in range(16):
                        for k in range(8):
                            row_idx = (i*16 + j)*8 + k
                            for r in range(27):
                                col_idx = k*27 + r
                                A_W[row_idx, col_idx] = U[i,r] * V[j,r]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
                W = W_flat.reshape(8, 27)
                W = np.round(W * 2) / 2
                
                # Check error
                err = verify_decomposition(U, V, W, T)
                if err < 1e-6:
                    return U, V, W
                    
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                    
                # Early stop if error is very small but not zero (might be stuck)
                if err < 1e-3:
                    # Try a few more iterations with tighter rounding?
                    # Or break to try another restart
                    break
                    
        return best_solution

    # Attempt to find rank 27
    result_27 = try_rank_reduction(rank_target=27, num_restarts=20, max_iters=50)
    
    if result_27 is not None:
        U_out, V_out, W_out = result_27
        # Final verification
        if verify_decomposition(U_out, V_out, W_out, T) < 1e-6:
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

    # If rank 27 not found, return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
