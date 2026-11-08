import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = 3, 4, 6

    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    # 2x3 Strassen: 2x4 * 4x3 -> 2x3 with 10 terms
    # A is 2x4, B is 4x3, C is 2x3
    # Split A into 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3)
    # Split B into 2x3 blocks: B_L (rows 0-1), B_R (rows 2-3)
    # C_L = A_L @ B_L, C_R = A_R @ B_R
    #
    # For 2x2 Strassen (7 terms) on each 2x2x3 subproblem:
    # Actually, let's use a direct 2x4x3 decomposition
    #
    # Better approach: Use the known 2x3 Strassen variant
    # Split B into columns: B = [B1 B2 B3] where each Bi is 4x1
    # Or split differently for efficiency
    #
    # Let me use a cleaner approach:
    # For 2x4 * 4x3:
    # A = [[a11,a12,a13,a14],[a21,a22,a23,a24]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33],[b41,b42,b43]]
    #
    # Using Strassen-style with 2x2 blocks:
    # M1 = (A_L) * (B_L) where A_L is 2x2, B_L is 2x3 -> need special handling
    #
    # Actually, let me use a different decomposition:
    # Group columns of A and rows of B into 2x2 blocks
    # A = [A11 A12] where A11, A12 are 2x2
    # B = [B11; B21] where B11, B21 are 2x3
    # C = A11*B11 + A12*B21
    #
    # For A11*B11 (2x2 * 2x3): 6 mults schoolbook or use Strassen trick
    # For A12*B21 (2x2 * 2x3): 6 mults schoolbook or use Strassen trick
    #
    # Better: Use the fact that we can share computations
    #
    # Let me try a direct approach with 10 terms for 2x4x3:
    s2x3_defs = [
        # Each term: (u_indices, v_indices, w_indices)
        # u is for 2x4 matrix, v is for 4x3 matrix, w is for 2x3 output
        #
        # Term 1: a11*b11 -> c11
        ([(0, 0, 1)], [(0, 0, 1)], [(0, 0, 1)]),
        # Term 2: a12*b21 -> c11
        ([(0, 1, 1)], [(1, 0, 1)], [(0, 0, 1)]),
        # Term 3: a13*b12 -> c12
        ([(0, 2, 1)], [(0, 1, 1)], [(0, 1, 1)]),
        # Term 4: a14*b22 -> c12
        ([(0, 3, 1)], [(1, 1, 1)], [(0, 1, 1)]),
        # Term 5: a13*b13 -> c13
        ([(0, 2, 1)], [(0, 2, 1)], [(0, 2, 1)]),
        # Term 6: a14*b23 -> c13
        ([(0, 3, 1)], [(1, 2, 1)], [(0, 2, 1)]),
        # Term 7: a21*b11 -> c21
        ([(1, 0, 1)], [(0, 0, 1)], [(1, 0, 1)]),
        # Term 8: a22*b21 -> c21
        ([(1, 1, 1)], [(1, 0, 1)], [(1, 0, 1)]),
        # Term 9: a23*b12 -> c22
        ([(1, 2, 1)], [(0, 1, 1)], [(1, 1, 1)]),
        # Term 10: a24*b22 -> c22
        ([(1, 3, 1)], [(1, 1, 1)], [(1, 1, 1)]),
    ]

    # This is just schoolbook with 12 terms. Let me use a true Strassen variant.
    #
    # For 2x4 * 4x3, we can use:
    # Split into 2x2 blocks: A = [A_L | A_R], B = [B_T; B_B]
    # A_L, A_R are 2x2, B_T, B_B are 2x3
    #
    # C = A_L @ B_T + A_R @ B_B
    #
    # For A_L @ B_T (2x2 * 2x3):
    #   Can use 7 Strassen terms if we pad to 2x2x2 and extend
    #   Or use a custom 2x2x3 decomposition
    #
    # Known: 2x2x3 matrix multiplication has rank 8 (not 7)
    # So 2x4x3 might have rank 8+8 = 16 with no sharing, or less with sharing
    #
    # Let me try a different approach: Use the existing 2x2x2 Strassen
    # and handle the extra column separately
    #
    # For 2x4 * 4x3:
    # C[:, 0:2] = A @ B[:, 0:2] (2x4 * 4x2 = 2x2) -> 7 Strassen terms
    # C[:, 2] = A @ B[:, 2] (2x4 * 4x1 = 2x1) -> 4 terms schoolbook
    # Total: 7 + 4 = 11 terms
    #
    # This is better than 12! Let's use this.

    s2x2_defs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    rank = 24 + 2 * 11  # Row 0: 24, Rows 1-2: 2 blocks * 11 = 22, total 46
    # Wait, let me recalculate:
    # For rows 1-2, we have 2x4 * 4x6
    # Split B into 2 blocks of 3 columns: B1 (cols 0-2), B2 (cols 3-5)
    # For each 2x4 * 4x3: 7 (Strassen for 2x2x2 part) + 4 (schoolbook for 2x2x1 part) = 11
    # Total for rows 1-2: 2 * 11 = 22
    # Grand total: 24 + 22 = 46

    rank = 46
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_entries:
            U[idx, col] += val
        for idx, val in v_entries:
            V[idx, col] += val
        for idx, val in w_entries:
            W[idx, col] += val
        col += 1

    # Part 1: Schoolbook for row 0 (24 terms)
    for j in range(m):
        for k in range(p):
            u = [(ui(0, j), 1)]
            v = [(vi(j, k), 1)]
            w = [(wi(k, 0), 1)]
            add_term(u, v, w)

    # Part 2: Strassen + schoolbook for rows 1-2
    # For each pair of B columns (3 at a time):
    for b_start in range(0, p, 3):
        # B_sub is 4x3, rows 0-3, cols b_start to b_start+2
        # A_sub is 2x4, rows 1-2, cols 0-3
        # C_sub is 2x3, rows 1-2, cols b_start to b_start+2

        # First 2 columns: 2x4 * 4x2 -> 2x2 using Strassen (7 terms)
        # Split A into left 2 cols and right 2 cols
        # Split B into top 2 rows and bottom 2 rows

        for a_c, b_c, c_c in s2x2_defs:
            u_entries, v_entries, w_entries = [], [], []
            # A indices: 2x2 block, row 0-1, col 0-1
            for idx, val in enumerate(a_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    u_entries.append((ui(1 + r, c), val))
            # B indices: 2x2 block, row 0-1, col 0-1 (of the 2-col subset)
            for idx, val in enumerate(b_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    v_entries.append((vi(r, b_start + c), val))
            # C indices: 2x2 block, row 0-1, col 0-1
            for idx, val in enumerate(c_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    w_entries.append((wi(b_start + c, 1 + r), val))
            add_term(u_entries, v_entries, w_entries)

        # Third column: 2x4 * 4x1 -> 2x1 using schoolbook (4 terms)
        for j in range(m):
            u = [(ui(1 + (j // 2), j % 2), 1)] if j < 2 else [(ui(1 + (j // 2), j % 2), 1)]
            # Actually, A is 2x4, so row 0 is A[1,:], row 1 is A[2,:]
            # For column j of A (j=0,1,2,3), row r (r=0,1):
            # U index = ui(1+r, j)
            u = [(ui(1, j) if j < 2 else ui(2, j - 2), 1)] if False else [(ui(1 + (j % 2), j // 2), 1)]
            # Let me be more careful:
            # A_sub has rows 1,2 and cols 0,1,2,3
            # For term (j, k) where k = b_start+2:
            # A[1+j//2, j%2] * B[j, b_start+2] -> C[1+j//2, b_start+2]
            # Wait, A is 2x4: A[0,:] = A[1,0:4] of original, A[1,:] = A[2,0:4]
            # So A_sub[i, j] = original A[1+i, j]
            # B_sub[j, k] = original B[j, b_start + k]
            # C_sub[i, k] = original C[1+i, b_start + k]

            u = [(ui(1 + (j % 2), j // 2), 1)]
            v = [(vi(j, b_start + 2), 1)]
            w = [(wi(b_start + 2, 1 + (j % 2)), 1)]
            add_term(u, v, w)

    # Trim zero columns
    mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
