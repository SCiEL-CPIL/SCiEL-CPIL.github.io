import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit

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

    # Fallback: schoolbook rank 32
    def get_fallback():
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

    if err_s > 1e-9:
        return get_fallback()

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
        return get_fallback()

    # NEW STRATEGY: Structured Algebraic Search for Rank 27
    # The tensor T is the sum of 4 disjoint 2x2x2 tensors.
    # Rank 28 is achieved by 4 independent Strassen blocks.
    # To get rank 27, we must "share" one rank-1 term between blocks or modify the structure.
    # We hypothesize that a rank-27 decomposition exists where the 28th term is a linear combination
    # of the others, or where the block structure is slightly perturbed to allow overlap.
    
    # We will search for a rank-27 solution by fixing U and V to have a specific "almost block-diagonal"
    # structure with a small perturbation that allows a single shared term.
    # Specifically, we assume the 27th term is a "cross-term" that interacts with multiple blocks.
    
    # We will use a constrained optimization approach:
    # 1. Start with the rank-28 solution.
    # 2. Propose a rank-27 structure where one column is removed and the remaining 27 columns are adjusted.
    # 3. Solve the linear system for the adjusted columns.
    # 4. Check if the solution has "nice" (integer/half-integer) entries.
    
    def try_algebraic_rank27():
        # We will try to remove one of the 28 terms and see if the remaining 27 can span the tensor.
        # This is equivalent to checking if the 28x28 matrix of coefficients is singular in a specific way.
        
        # Instead of random removal, we try to "merge" two terms into one by linear combination.
        # Let's try to express the 28th term as a linear combination of the first 27.
        # This is a system of linear equations: T_28 = sum_{r=1}^{27} c_r T_r
        
        # We will solve for the coefficients c_r.
        # If a solution exists, we can combine T_28 and T_r into a single term.
        
        # Let's build the matrix of tensor entries for the 28 terms.
        # Each term T_r is a 8x16x8 tensor (flattened to 1024x1).
        # We want to see if T_28 is in the span of {T_1, ..., T_27}.
        
        # Flatten the tensors
        T_flat = T.flatten()
        
        # Build the matrix of the first 27 terms
        # U28[:, r], V28[:, r], W28[:, r] define the r-th term
        # Term r is outer product U[:,r] V[:,r] W[:,r]
        
        # We will solve for c such that T = sum_{r=1}^{27} c_r * (U[:,r] V[:,r] W[:,r])
        # This is a linear system: A * c = T_flat
        # where A[:, r] = (U[:,r] * V[:,r]) flattened, but wait, the third mode is W.
        # Actually, T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For a fixed r, the tensor is U[:,r] \otimes V[:,r] \otimes W[:,r]
        
        # We want to check if T can be represented with rank 27.
        # We fix U and V to be the first 27 columns of U28 and V28.
        # Then we solve for W.
        
        # Let's try removing each of the 28 terms and see if the rest can span T.
        for remove_idx in range(28):
            if time.time() - start_time > deadline - 10:
                return None
            
            # Keep all terms except remove_idx
            U_red = np.delete(U28, remove_idx, axis=1)
            V_red = np.delete(V28, remove_idx, axis=1)
            
            # Solve for W_red
            # T[i,j,k] = sum_{r=0}^{26} U_red[i,r] V_red[j,r] W_red[k,r]
            # For each k, T[:,:,k] = sum_r W_red[k,r] (U_red[:,r] V_red[:,r]^T)
            
            W_red = np.zeros((8, 27))
            valid = True
            
            for k in range(8):
                T_k_vec = T[:, :, k].flatten()
                A_k = np.zeros((128, 27))
                for r in range(27):
                    A_k[:, r] = np.outer(U_red[:, r], V_red[:, r]).flatten()
                
                W_k, residuals, rank, sv = np.linalg.lstsq(A_k, T_k_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-5:
                    valid = False
                    break
                W_red[k, :] = W_k
            
            if not valid:
                continue
            
            # Verify
            err = verify_decomposition(U_red, V_red, W_red, T)
            if err < 1e-5:
                # Check for nice entries in W_red
                W_rounded = np.round(W_red * 2) / 2
                err_rounded = verify_decomposition(U_red, V_red, W_rounded, T)
                if err_rounded < 1e-5:
                    return U_red, V_red, W_rounded
        
        return None

    result_27 = try_algebraic_rank27()
    
    if result_27 is not None:
        U_out, V_out, W_out = result_27
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        
    # If rank 27 not found, return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
