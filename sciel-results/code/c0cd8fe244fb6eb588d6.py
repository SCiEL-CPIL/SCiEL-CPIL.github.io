"""Combined program: block-tiling (Strassen 2x2x2 + trivial 1x2x2) for 3x4x4."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = 3, 4, 4

    # We tile the 3x4x4 multiplication into 4 independent 3x2x2 blocks.
    # Each 3x2x2 block: A is 3x2, B is 2x2, C is 2x3.
    # We decompose each 3x2x2 block into:
    #   - A 2x2x2 sub-block (rows 0,1 of A) using Strassen (rank 7)
    #   - A 1x2x2 sub-block (row 2 of A) using trivial (rank 4)
    # Total rank per block = 7 + 4 = 11
    # Total rank = 4 * 11 = 44

    rank_total = 44
    U = np.zeros((n * m, rank_total))  # (12, 44)
    V = np.zeros((m * p, rank_total))  # (16, 44)
    W = np.zeros((p * n, rank_total))  # (12, 44)

    # Strassen decomposition for 2x2x2
    # U_loc: (4, 7), V_loc: (4, 7), W_loc: (4, 7)
    # Local indices: A[0,1,2,3] = A[0,0],A[0,1],A[1,0],A[1,1]
    #                B[0,1,2,3] = B[0,0],B[0,1],B[1,0],B[1,1]
    #                C[0,1,2,3] = C[0,0],C[0,1],C[1,0],C[1,1]
    def strassen_2x2x2():
        Ul = np.zeros((4, 7))
        Vl = np.zeros((4, 7))
        Wl = np.zeros((4, 7))
        # M1 = (A00+A11)*(B00+B11)
        Ul[0,0]=1; Ul[3,0]=1
        Vl[0,0]=1; Vl[3,0]=1
        Wl[0,0]=1; Wl[3,0]=1
        # M2 = (A10+A11)*B00
        Ul[2,1]=1; Ul[3,1]=1
        Vl[0,1]=1
        Wl[2,1]=1; Wl[3,1]=-1
        # M3 = A00*(B01-B11)
        Ul[0,2]=1
        Vl[1,2]=1; Vl[3,2]=-1
        Wl[1,2]=1; Wl[3,2]=1
        # M4 = A11*(B10-B00)
        Ul[3,3]=1
        Vl[2,3]=1; Vl[0,3]=-1
        Wl[0,3]=1; Wl[2,3]=1
        # M5 = (A00+A01)*B11
        Ul[0,4]=1; Ul[1,4]=1
        Vl[3,4]=1
        Wl[0,4]=-1; Wl[1,4]=1
        # M6 = (A10-A00)*(B00+B01)
        Ul[2,5]=1; Ul[0,5]=-1
        Vl[0,5]=1; Vl[1,5]=1
        Wl[3,5]=1
        # M7 = (A01-A11)*(B10+B11)
        Ul[1,6]=1; Ul[3,6]=-1
        Vl[2,6]=1; Vl[3,6]=1
        Wl[0,6]=1
        return Ul, Vl, Wl

    # Trivial 1x2x2 decomposition (rank 4)
    # A is 1x2 (single row, 2 cols), B is 2x2, C is 2x1 (2 rows, 1 col)
    # C[k] = sum_j A[j] * B[j,k] for k=0,1
    # This is just 4 scalar products:
    # term0: A[0]*B[0,0] -> C[0]
    # term1: A[1]*B[1,0] -> C[0]
    # term2: A[0]*B[0,1] -> C[1]
    # term3: A[1]*B[1,1] -> C[1]
    def trivial_1x2x2():
        Ul = np.zeros((2, 4))  # A has 1 row x 2 cols = 2 elements
        Vl = np.zeros((4, 4))  # B is 2x2 = 4 elements
        Wl = np.zeros((2, 4))  # C is 2x1 = 2 elements
        # term0: A[0] * B[0,0] -> C[0]
        Ul[0,0]=1; Vl[0,0]=1; Wl[0,0]=1
        # term1: A[1] * B[1,0] -> C[0]
        Ul[1,1]=1; Vl[2,1]=1; Wl[0,1]=1
        # term2: A[0] * B[0,1] -> C[1]
        Ul[0,2]=1; Vl[1,2]=1; Wl[1,2]=1
        # term3: A[1] * B[1,1] -> C[1]
        Ul[1,3]=1; Vl[3,3]=1; Wl[1,3]=1
        return Ul, Vl, Wl

    Ul_s, Vl_s, Wl_s = strassen_2x2x2()
    Ul_t, Vl_t, Wl_t = trivial_1x2x2()

    # Now tile 4 blocks. Each block handles columns 2*block_idx and 2*block_idx+1 of A,
    # and the corresponding 2x2 sub-blocks of B and C.
    # A index: i*m + j where i in [0,3), j in [0,4)
    # B index: j*p + k where j in [0,4), k in [0,4)
    # C index: k*n + i where k in [0,4), i in [0,3)
    #
    # Block b (b=0,1,2,3) handles:
    #   A columns: 2b, 2b+1  (so A rows i, cols 2b and 2b+1)
    #   B rows: 2b, 2b+1; all 4 cols of B? No.
    #
    # Wait, let me reconsider the tiling.
    # A is 3x4, B is 4x4, C is 4x3.
    # We want to split B's 4 columns into 2 groups of 2, and A's 4 columns into 2 groups of 2?
    # Actually, the natural tiling is:
    # C[k, i] = sum_j A[i, j] * B[j, k]
    # We can split the sum over j into j in {0,1}, {2,3}, etc. But B is 4x4, so j goes 0..3.
    #
    # Alternative: split A's columns into 2 groups of 2: j in {0,1} and j in {2,3}.
    # And split B's rows correspondingly: B rows {0,1} and {2,3}.
    # Then C = A_left * B_left + A_right * B_right, where A_left is 3x2, B_left is 2x4, etc.
    # But B_left is 2x4, not 2x2. So we'd need to further split B's columns.
    #
    # Let me think differently. The 3x2x2 block means:
    # A_block: 3x2 (3 rows, 2 cols)
    # B_block: 2x2 (2 rows, 2 cols)
    # C_block: 2x3 (2 rows, 3 cols)
    #
    # This corresponds to: for a fixed pair of A-columns (j, j+1) and a fixed pair of B-rows (j, j+1) and a fixed pair of C-rows (k, k+1):
    # C[k, i] += sum_{j in {j0, j0+1}} A[i, j] * B[j, k]
    #
    # So we need 2 tiles in the j-direction (to cover 4 cols of A and 4 rows of B) and 2 tiles in the k-direction (to cover 4 rows of C).
    # That's 2*2 = 4 blocks, each 3x2x2. Total rank 4*11 = 44.

    for block_j in range(2):  # covers A cols 2*block_j, 2*block_j+1 and B rows 2*block_j, 2*block_j+1
        for block_k in range(2):  # covers C rows 2*block_k, 2*block_k+1
            base = (block_j * 2 + block_k) * 11

            # A indices: i in [0,3), j in {2*block_j, 2*block_j+1}
            # A index = i*4 + j
            # B indices: j in {2*block_j, 2*block_j+1}, k in {2*block_k, 2*block_k+1}
            # B index = j*4 + k
            # C indices: k in {2*block_k, 2*block_k+1}, i in [0,3)
            # C index = k*3 + i

            # Strassen part: A rows i=0,1 (local 0,1), cols j=2*block_j, 2*block_j+1 (local 0,1)
            # B rows j=2*block_j, 2*block_j+1 (local 0,1), cols k=2*block_k, 2*block_k+1 (local 0,1)
            # C rows k=2*block_k, 2*block_k+1 (local 0,1), cols i=0,1 (local 0,1)

            for r in range(7):
                # U
                for i_local in range(2):
                    if Ul_s[i_local, r] != 0:
                        i_global = i_local
                        j_global = 2 * block_j + (1 if i_local == 1 else 0)  # Wait, A is 3x2, so local col 0 -> global col 2*block_j, local col 1 -> global col 2*block_j+1
                        # Actually, for the 2x2 sub-block, A[i_local, col_local] where col_local in {0,1}
                        # The 2x2x2 Strassen uses A[0,0], A[0,1], A[1,0], A[1,1]
                        # So local index: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
                        # i_local is the row, and we need to figure out the col from the Strassen pattern
                        pass

            # Let me redo this more carefully.
            # In the Strassen 2x2x2, the local A indices are:
            # 0: A[0,0], 1: A[0,1], 2: A[1,0], 3: A[1,1]
            # So for local index la: row = la // 2, col = la % 2
            # Global: row = i (0 or 1), col = 2*block_j + col_local
            # A global index = row * 4 + col

            # In the Strassen 2x2x2, the local B indices are:
            # 0: B[0,0], 1: B[0,1], 2: B[1,0], 3: B[1,1]
            # For local index lb: row = lb // 2, col = lb % 2
            # Global: row = 2*block_j + row_local, col = 2*block_k + col_local
            # B global index = row * 4 + col

            # In the Strassen 2x2x2, the local C indices are:
            # 0: C[0,0], 1: C[0,1], 2: C[1,0], 3: C[1,1]
            # For local index lc: row = lc // 2, col = lc % 2
            # Global: row = 2*block_k + row_local, col = i (0 or 1)
            # C global index = row * 3 + col

            for r in range(7):
                # U contributions
                for la in range(4):
                    if Ul_s[la, r] != 0:
                        row_local = la // 2
                        col_local = la % 2
                        a_idx = row_local * 4 + (2 * block_j + col_local)
                        U[a_idx, base + r] += Ul_s[la, r]
                # V contributions
                for lb in range(4):
                    if Vl_s[lb, r] != 0:
                        row_local = lb // 2
                        col_local = lb % 2
                        b_idx = (2 * block_j + row_local) * 4 + (2 * block_k + col_local)
                        V[b_idx, base + r] += Vl_s[lb, r]
                # W contributions
                for lc in range(4):
                    if Wl_s[lc, r] != 0:
                        row_local = lc // 2
                        col_local = lc % 2
                        c_idx = (2 * block_k + row_local) * 3 + col_local
                        W[c_idx, base + r] += Wl_s[lc, r]

            # Trivial 1x2x2 part: A row i=2, cols j=2*block_j, 2*block_j+1
            # B rows j=2*block_j, 2*block_j+1, cols k=2*block_k, 2*block_k+1
            # C rows k=2*block_k, 2*block_k+1, col i=2
            # Local A: 2 elements (A[2, 2*block_j], A[2, 2*block_j+1])
            # Local B: 4 elements
            # Local C: 2 elements (C[2*block_k, 2], C[2*block_k+1, 2])

            for r in range(4):
                # U contributions
                for la in range(2):
                    if Ul_t[la, r] != 0:
                        col_local = la
                        a_idx = 2 * 4 + (2 * block_j + col_local)
                        U[a_idx, base + 7 + r] += Ul_t[la, r]
                # V contributions
                for lb in range(4):
                    if Vl_t[lb, r] != 0:
                        row_local = lb // 2
                        col_local = lb % 2
                        b_idx = (2 * block_j + row_local) * 4 + (2 * block_k + col_local)
                        V[b_idx, base + 7 + r] += Vl_t[lb, r]
                # W contributions
                for lc in range(2):
                    if Wl_t[lc, r] != 0:
                        row_local = lc
                        c_idx = (2 * block_k + row_local) * 3 + 2
                        W[c_idx, base + 7 + r] += Wl_t[lc, r]

    # Verify
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                if j == j2 and i == i3 and k == k2:
                                    continue
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U0 = np.zeros((n * m, rank0))
        V0 = np.zeros((m * p, rank0))
        W0 = np.zeros((p * n, rank0))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col] = 1
                    V0[j * p + k, col] = 1
                    W0[k * n + i, col] = 1
                    col += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
