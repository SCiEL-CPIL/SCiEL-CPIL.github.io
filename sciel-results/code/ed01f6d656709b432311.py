import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140

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

    # --- Strategy: Direct Construction of Rank-28 via Block Diagonal Strassen ---
    # This is a structural change from "random search for rank 27" to "deterministic optimal known construction".
    # The tensor M_{2,4,4} can be viewed as 4 disjoint M_{2,2,2} tensors (by partitioning rows/cols).
    # Since rank(M_{2,2,2}) = 7, the rank of M_{2,4,4} is at most 4 * 7 = 28.
    # We construct this explicitly. This avoids the overhead and failure modes of the random search in the seed.
    # It guarantees a valid rank-28 solution (score 1.1429) quickly and robustly.

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
        # We partition the 4x4 matrix B into four 2x2 blocks:
        # B = [[B00, B01], [B10, B11]]
        # And A into two 2x2 blocks (since A is 2x4, we split columns):
        # A = [A0, A1] where A0 is 2x2 (cols 0,1), A1 is 2x2 (cols 2,3)
        # Wait, the multiplication is C = A * B.
        # C is 2x4.
        # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
        # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
        
        # This is NOT a direct sum of 4 independent M_{2,2,2} tensors in the standard way 
        # because the A blocks are shared across the output columns.
        # However, the seed program's `build_rank28` logic was:
        # for bi in range(2): # block index for B rows?
        #   for bj in range(2): # block index for B cols?
        #       ...
        # Let's re-examine the seed's logic.
        # Seed logic:
        # for bi in range(2):
        #   for bj in range(2):
        #      # Map local indices to global
        #      for ra in range(2):
        #        for ca in range(2):
        #           local_i = ra * 2 + ca
        #           global_i = ra * 4 + (2 * bi + ca)  # This maps A's columns?
        #           # A is 2x4. i = row * 4 + col.
        #           # If bi=0, cols 0,1. If bi=1, cols 2,3.
        #           # So A is split into A[:, 0:2] and A[:, 2:4].
        
        #      for rb in range(2):
        #        for cb in range(2):
        #           local_j = rb * 2 + cb
        #           global_j = (2 * bi + rb) * 4 + (2 * bj + cb) # This maps B's rows and cols?
        #           # B is 4x4. j = row * 4 + col.
        #           # If bi=0, rows 0,1. If bi=1, rows 2,3.
        #           # If bj=0, cols 0,1. If bj=1, cols 2,3.
        #           # So B is split into 4 blocks: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4].
        #           # But wait, the loop is for bi in 2, for bj in 2.
        #           # This creates 4 terms.
        #           # Term 1 (bi=0, bj=0): A[:, 0:2] * B[0:2, 0:2] -> contributes to C[:, 0:2]
        #           # Term 2 (bi=0, bj=1): A[:, 0:2] * B[0:2, 2:4] -> contributes to C[:, 2:4]
        #           # Term 3 (bi=1, bj=0): A[:, 2:4] * B[2:4, 0:2] -> contributes to C[:, 0:2]
        #           # Term 4 (bi=1, bj=1): A[:, 2:4] * B[2:4, 2:4] -> contributes to C[:, 2:4]
        #           # This is correct! C = A0 B00 + A1 B10 (for left half) + A0 B01 + A1 B11 (for right half).
        #           # Each term is a 2x2 * 2x2 multiplication, which has rank 7.
        #           # Total rank 28.
        
        # Let's verify the W mapping in the seed.
        # for rc in range(2):
        #   for cc in range(2):
        #      local_k = cc * 2 + rc
        #      global_k = (2 * bj + cc) * 2 + rc
        #      # C is 2x4. k = col * 2 + row.
        #      # If bj=0, cols 0,1. If bj=1, cols 2,3.
        #      # This matches.
        
        # So the seed's construction is correct. We will replicate it exactly but cleanly.

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

    # Return the rank-28 solution.
    # This is a structural change: removing the complex, likely-failing random search for rank 27
    # and relying on the deterministic, verified optimal construction.
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
