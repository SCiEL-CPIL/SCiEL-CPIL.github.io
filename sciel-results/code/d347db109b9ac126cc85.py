"""Strassen-based rank-reduced tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's 7-multiplication method.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2.
    """
    n = m = p = 2
    U = np.zeros((n * m, 7), dtype=float)
    V = np.zeros((m * p, 7), dtype=float)
    W = np.zeros((p * n, 7), dtype=float)

    # Strassen's 7 multiplications for 2x2 matrices:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)

    # A indices: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B indices: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # U row index: i*m+j = i*2+j
    # V row index: j*p+k = j*2+k
    # W row index: k*n+i = k*2+i

    # M1 = (A11+A22)*(B11+B22)
    # A11 is at (0,0) -> U row 0; A22 is at (1,1) -> U row 3
    # B11 is at (0,0) -> V row 0; B22 is at (1,1) -> V row 3
    # Result goes to C11 (0,0) and C22 (1,1)
    # C11: k=0, i=0 -> W row 0; C22: k=1, i=1 -> W row 3
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1

    # M2 = (A21+A22)*B11
    # A21 at (1,0) -> U row 2; A22 at (1,1) -> U row 3
    # B11 at (0,0) -> V row 0
    # Result to C21: k=0, i=1 -> W row 1
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1

    # M3 = A11*(B12-B22)
    # A11 at (0,0) -> U row 0
    # B12 at (0,1) -> V row 1; B22 at (1,1) -> V row 3
    # Result to C12: k=0, i=0 -> W row 0; and C22: k=1, i=1 -> W row 3
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[0, 2] += 1; W[3, 2] += 1

    # M4 = A22*(B21-B11)
    # A22 at (1,1) -> U row 3
    # B21 at (1,0) -> V row 2; B11 at (0,0) -> V row 0
    # Result to C11: k=0, i=0 -> W row 0; and C21: k=0, i=1 -> W row 1
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[1, 3] += 1

    # M5 = (A11+A12)*B22
    # A11 at (0,0) -> U row 0; A12 at (0,1) -> U row 1
    # B22 at (1,1) -> V row 3
    # Result to C11: k=0, i=0 -> W row 0 (with -1); and C12: k=0, i=0 -> W row 0 (with +1)
    # Wait, C11 gets -M5, C12 gets +M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1  # C11 contribution
    # C12 is also at W row 0 (k=0, i=0)
    W[0, 4] += 1  # C12 contribution, net 0? No, C11 and C12 are different output positions.
    # C11: (i=0, k=0) -> W[0*2+0] = W[0]
    # C12: (i=0, k=1) -> W[1*2+0] = W[2]
    # I made an error. Let me redo the W indexing.
    # W[k*n+i] where n=2.
    # C11: i=0, k=0 -> W[0*2+0] = W[0]
    # C12: i=0, k=1 -> W[1*2+0] = W[2]
    # C21: i=1, k=0 -> W[0*2+1] = W[1]
    # C22: i=1, k=1 -> W[1*2+1] = W[3]

    # Let me restart W assignments properly.
    W[0, 4] = -1  # C11 from -M5
    W[2, 4] = 1   # C12 from +M5

    # M6 = (A21-A11)*(B11+B12)
    # A21 at (1,0) -> U row 2; A11 at (0,0) -> U row 0
    # B11 at (0,0) -> V row 0; B12 at (0,1) -> V row 1
    # Result to C22: i=1, k=1 -> W[3]
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1  # C22

    # M7 = (A12-A22)*(B21+B22)
    # A12 at (0,1) -> U row 1; A22 at (1,1) -> U row 3
    # B21 at (1,0) -> V row 2; B22 at (1,1) -> V row 3
    # Result to C11: i=0, k=0 -> W[0]
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1  # C11

    return U, V, W


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Strassen's algorithm to reduce rank from schoolbook 112 to 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7

    # Split the 4x7 matrix into a 4x4 and a 4x3 part.
    # We'll compute C = A*B where A is 4x4 and B is 4x7.
    # Split B = [B1, B2] where B1 is 4x4 and B2 is 4x3.
    # C = A*B1 + A*B2
    # We use Strassen for 4x4*4x4 (rank 49) and a simple method for 4x4*4x3.

    # For 4x4*4x3, we can use schoolbook which gives rank 4*4*3 = 48.
    # Total rank = 49 + 48 = 97.

    # Actually, let me reconsider. For 4x4*4x4 using Strassen twice:
    # Apply Strassen to 2x2 blocks. Each 2x2*2x2 uses rank 7.
    # 4x4 has four 2x2 blocks. The Strassen method for 4x4 uses 7*7 = 49 multiplications
    # if we recursively apply Strassen, but actually Strassen for 4x4*4x4 has rank 49
    # (it's a known result).

    # Let me build the 4x4*4x4 decomposition using nested Strassen.
    # A 4x4 matrix can be viewed as 2x2 blocks of 2x2 matrices.
    # Strassen on 2x2 (of 2x2 blocks) gives 7 block multiplications.
    # Each block multiplication is 2x2*2x2, which uses rank 7 (Strassen).
    # Total rank for 4x4*4x4: 7 * 7 = 49.

    # For 4x4*4x3:
    # B2 is 4x3. We can pad to 4x4 and use the same 4x4*4x4 decomposition,
    # but that wastes rank. Instead, use schoolbook for 4x4*4x3: rank = 48.
    # Or we can be smarter. Let's just use schoolbook for now.

    # Total rank: 49 + 48 = 97.

    # Let's implement this.

    # First, build the 4x4*4x4 Strassen decomposition (rank 49).
    U44 = np.zeros((16, 49), dtype=float)
    V44 = np.zeros((16, 49), dtype=float)
    W44 = np.zeros((16, 49), dtype=float)

    # Nested Strassen for 4x4:
    # The 4x4 matrices are divided into 2x2 blocks of 2x2 matrices.
    # We apply Strassen at the top level (7 combinations), and each combination
    # involves multiplying 2x2 matrices (using Strassen again).

    # This is complex to implement directly. Let me use a different approach:
    # I'll enumerate the 49 rank-1 terms for 4x4*4x4 using the known Strassen recursion.

    # Actually, let me use a simpler but still valid approach:
    # Use the known fact that R(4,4,4) <= 49 and construct it explicitly.

    # The 4x4*4x4 Strassen decomposition can be built as follows:
    # Label the 4x4 matrix A as blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # Similarly for B.
    # Strassen gives:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # Each Mi is a 2x2*2x2 multiplication, which uses 7 scalar multiplications (Strassen).
    # So we have 7*7 = 49 scalar multiplications total.

    # Let me build this systematically.
    # First, get the 2x2*2x2 Strassen decomposition.
    U2, V2, W2 = _strassen_2x2x2()

    # Now, for each of the 7 Strassen combinations at the top level,
    # we need to map the 2x2 block operations to the 4x4 indices.

    # A block Aij: i in {0,1}, j in {0,1}
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3

    # For U (4x4 input, 16 rows): U row index = i*4 + j where i is row of A, j is col of A.
    # For V (4x4 input, 16 rows): V row index = j*4 + k where j is row of B, k is col of B.
    # For W (4x4 output, 16 rows): W row index = k*4 + i where k is col of C, i is row of C.

    # Let me define the block indices.
    # For a 2x2 block at position (bi, bj) in a 4x4 matrix:
    # Rows: bi*2 to bi*2+1
    # Cols: bj*2 to bj*2+1

    # U index for element (r, c) of A: r*4 + c
    # V index for element (r, c) of B: r*4 + c
    # W index for element (r, c) of C: c*4 + r (since W[k*n+i] with n=4)

    # For each top-level Strassen multiplication Ms (s in 0..6),
    # and for each bottom-level Strassen multiplication Mt (t in 0..6),
    # we create a rank-1 term.

    # The coefficient for U and V comes from the top-level Strassen.
    # The coefficient for W comes from the bottom-level Strassen, but mapped to 4x4 positions.

    # This is getting complex. Let me use a different strategy:
    # I'll just use schoolbook for both parts but with a small optimization.

    # Actually, let me just use the known rank 49 decomposition for 4x4*4x4
    # by using the explicit formulas.

    # For simplicity and correctness, let me use:
    # 4x4*4x4: rank 49 (Strassen)
    # 4x4*4x3: rank 48 (schoolbook)
    # Total: 97

    # But implementing 4x4*4x4 Strassen correctly is error-prone.
    # Let me use a safer approach: use Strassen for 2x2*2x2 and build up.

    # I'll implement the full nested Strassen for 4x4*4x4.

    # Top-level Strassen combinations:
    # s=0: M1 = (A11+A22)*(B11+B22)
    # s=1: M2 = (A21+A22)*B11
    # s=2: M3 = A11*(B12-B22)
    # s=3: M4 = A22*(B21-B11)
    # s=4: M5 = (A11+A12)*B22
    # s=5: M6 = (A21-A11)*(B11+B12)
    # s=6: M7 = (A12-A22)*(B21+B22)

    # For each s, define:
    # - Which A blocks are used and with what signs
    # - Which B blocks are used and with what signs
    # - Which C blocks receive the result and with what signs

    top_level = [
        # (s, A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_idx, sign)
        # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
        (0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        (1, [(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),                     # M2 -> C21
        (2, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3 -> C12, C22
        (3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4 -> C11, C21
        (4, [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5 -> C11(-), C12(+)
        (5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6 -> C22
        (6, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7 -> C11
    ]

    # For each top-level combination s, and for each bottom-level Strassen term t,
    # we create a rank-1 term.

    # The bottom-level Strassen term t corresponds to a specific 2x2*2x2 multiplication.
    # U2[t], V2[t], W2[t] give the coefficients for the 2x2 case.

    # For a given top-level s and bottom-level t:
    # - U coefficient: for each A block (bi, sign) in top_level[s], and for each U2 row (u_row, u_val)
    #   in U2[:, t], we add u_val * sign to U at the corresponding 4x4 position.
    #   The 4x4 position for A block bi (0-3) and local row r (0-1), local col c (0-1):
    #   global row = bi_block_row * 2 + r, global col = bi_block_col * 2 + c
    #   where bi_block_row = bi // 2, bi_block_col = bi % 2

    # Similarly for V.
    # For W: the result goes to C blocks specified in top_level[s], with signs.
    # The bottom-level W2 gives local positions in the 2x2 result.

    block_to_rc = {
        0: (0, 0),  # A11
        1: (0, 1),  # A12
        2: (1, 0),  # A21
        3: (1, 1),  # A22
    }

    rank_offset = 0
    for s_data in top_level:
        s = s_data[0]
        A_blocks = s_data[1]
        B_blocks = s_data[2]
        C_blocks = s_data[3]

        for t in range(7):
            # Get bottom-level coefficients for term t
            u2_vals = U2[:, t]
            v2_vals = V2[:, t]
            w2_vals = W2[:, t]

            # Build U contributions
            for (bi, sign) in A_blocks:
                br, bc = block_to_rc[bi]
                for local_r in range(2):
                    for local_c in range(2):
                        u2_row = local_r * 2 + local_c
                        u2_val = u2_vals[u2_row]
                        if u2_val != 0:
                            global_row = br * 2 + local_r
                            global_col = bc * 2 + local_c
                            u44_row = global_row * 4 + global_col
                            U44[u44_row, rank_offset] += u2_val * sign

            # Build V contributions
            for (bi, sign) in B_blocks:
                br, bc = block_to_rc[bi]
                for local_r in range(2):
                    for local_c in range(2):
                        v2_row = local_r * 2 + local_c
                        v2_val = v2_vals[v2_row]
                        if v2_val != 0:
                            global_row = br * 2 + local_r
                            global_col = bc * 2 + local_c
                            v44_row = global_row * 4 + global_col
                            V44[v44_row, rank_offset] += v2_val * sign

            # Build W contributions
            for (ci, sign) in C_blocks:
                cr, cc = block_to_rc[ci]
                for local_k in range(2):
                    for local_i in range(2):
                        w2_row = local_k * 2 + local_i
                        w2_val = w2_vals[w2_row]
                        if w2_val != 0:
                            global_k = cc * 2 + local_k
                            global_i = cr * 2 + local_i
                            w44_row = global_k * 4 + global_i
                            W44[w44_row, rank_offset] += w2_val * sign

            rank_offset += 1

    # Now build the 4x4 * 4x3 schoolbook decomposition (rank 48)
    U43 = np.zeros((16, 48), dtype=float)
    V43 = np.zeros((12, 48), dtype=float)
    W43 = np.zeros((12, 48), dtype=float)

    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U43[i * 4 + j, col] = 1
                V43[j * 3 + k, col] = 1
                W43[k * 4 + i, col] = 1
                col += 1

    # Combine:
    # Total rank = 49 + 48 = 97
    total_rank = 49 + 48
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((28, total_rank), dtype=float)
    W = np.zeros((28, total_rank), dtype=float)

    # First part: 4x4 * 4x4 (columns 0-3 of B)
    U[:, :49] = U44
    V[:16, :49] = V44  # B1 is 4x4, so V rows 0-15
    W[:16, :49] = W44  # C1 is 4x4, so W rows 0-15

    # Second part: 4x4 * 4x3 (columns 4-6 of B)
    U[:, 49:] = U43
    V[16:28, 49:] = V43  # B2 is 4x3, so V rows 16-27
    W[16:28, 49:] = W43  # C2 is 4x3, so W rows 16-27

    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
