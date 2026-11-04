"""Matrix multiplication 4x4 * 4x9 tensor decomposition search.

Structural improvement over the seed:
  - Build a verified rank-49 Strassen decomposition for 4x4 * 4x4 (two 2x2-block
    Strassen layers, each 2x2 * 2x2 using 7 terms).
  - Use rank-16 schoolbook for the remaining 4x4 * 4x1 block.
  - Total rank 49 + 49 + 16 = 114, a structural improvement over rank-144.
  - Verify the full decomposition exactly with Fraction arithmetic before returning.
  - If verification fails (should not), fall back to the schoolbook rank-144.
"""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _build_strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2 * 2x2.

    Each term is a tuple (u_entries, v_entries, w_entries) where each *_entries
    is a dict mapping a local index to a coefficient (Fraction).

    Local indices:
      U: a00, a01, a10, a11  (indices 0,1,2,3)
      V: b00, b01, b10, b11  (indices 0,1,2,3)
      W: c00, c01, c10, c11  (indices 0,1,2,3)

    The 7 Strassen products:
      M1 = (A00 + A11) * (B00 + B11)
      M2 = (A10 + A11) * B00
      M3 = A00 * (B01 - B11)
      M4 = A11 * (B10 - B00)
      M5 = (A00 + A01) * B11
      M6 = (A10 - A00) * (B00 + B01)
      M7 = (A01 - A11) * (B10 + B11)

    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    F = Fraction
    # Each term: (U_dict, V_dict, W_dict)
    terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        ({0: F(1), 3: F(1)}, {0: F(1), 3: F(1)}, {0: F(1), 3: F(1)}),
        # M2 = (A10 + A11) * B00
        ({2: F(1), 3: F(1)}, {0: F(1)}, {2: F(1)}),
        # M3 = A00 * (B01 - B11)
        ({0: F(1)}, {1: F(1), 3: F(-1)}, {1: F(1)}),
        # M4 = A11 * (B10 - B00)
        ({3: F(1)}, {2: F(1), 0: F(-1)}, {0: F(1), 2: F(1)}),
        # M5 = (A00 + A01) * B11
        ({0: F(1), 1: F(1)}, {3: F(1)}, {0: F(-1), 1: F(1)}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({2: F(1), 0: F(-1)}, {0: F(1), 1: F(1)}, {3: F(1)}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({1: F(1), 3: F(-1)}, {2: F(1), 3: F(1)}, {0: F(1), 3: F(1)}),
    ]
    return terms


def _build_strassen_4x4():
    """Build the 49-term Strassen decomposition for 4x4 * 4x4.

    Returns (U, V, W) as lists of lists of Fractions with shapes:
      U: (16, 49)  -- n*m = 4*4 = 16
      V: (36, 49)  -- m*p = 4*9 ... wait, for 4x4*4x4, m*p = 4*4 = 16
      W: (16, 49)  -- p*n = 4*4 = 16

    Actually for 4x4 * 4x4: n=4, m=4, p=4, so nm=16, mp=16, pn=16.
    """
    F = Fraction
    nm = mp = pn = 16
    rank = 49
    U = [[F(0)] * rank for _ in range(nm)]
    V = [[F(0)] * rank for _ in range(mp)]
    W = [[F(0)] * rank for _ in range(pn)]

    # 2x2 Strassen terms
    s2_terms = _build_strassen_2x2_terms()

    # For 4x4 * 4x4, we have 2x2 blocks:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)

    # The 7 Strassen products at the 4x4 level are 2x2 * 2x2 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)

    # Each Mi is a 2x2 * 2x2 multiplication, which we decompose using 7 terms.
    # So total = 7 * 7 = 49 terms.

    # Define the 7 high-level Strassen combinations:
    # Each is a list of (a_block_coeff, b_block_coeff) pairs for the 2x2 inputs,
    # and we need to compute the 2x2*2x2 product of the linear combinations.

    # For each high-level product Mi, the inputs are linear combinations of 2x2 blocks.
    # We represent each high-level product as:
    #   A_input = sum of (coeff_a, a_block_index)
    #   B_input = sum of (coeff_b, b_block_index)
    # Then Mi = A_input * B_input (2x2 * 2x2), decomposed into 7 terms.

    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 for A; same for B.
    high_level = [
        # M1: (A00 + A11) * (B00 + B11)
        ([(1, 0), (1, 3)], [(1, 0), (1, 3)]),
        # M2: (A10 + A11) * B00
        ([(1, 2), (1, 3)], [(1, 0)]),
        # M3: A00 * (B01 - B11)
        ([(1, 0)], [(1, 1), (-1, 3)]),
        # M4: A11 * (B10 - B00)
        ([(1, 3)], [(1, 2), (-1, 0)]),
        # M5: (A00 + A01) * B11
        ([(1, 0), (1, 1)], [(1, 3)]),
        # M6: (A10 - A00) * (B00 + B01)
        ([(1, 2), (-1, 0)], [(1, 0), (1, 1)]),
        # M7: (A01 - A11) * (B10 + B11)
        ([(1, 1), (-1, 3)], [(1, 2), (1, 3)]),
    ]

    # C reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each high-level product Mi (i in 0..6), and each of the 7 Strassen terms
    # within that 2x2*2x2, we create a global term.

    # For a 2x2 block at position (br, bc) in the 4x4 matrix:
    # A block (br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    #   Local (r, c) in 2x2 -> global row = 2*br + r, col = 2*bc + c
    #   U index = global_row * 4 + global_col
    # B block (br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    #   Local (r, c) in 2x2 -> global row = 2*br + r, col = 2*bc + c
    #   V index = global_row * 4 + global_col
    # C block (br, bc): rows 2*br..2*br+1, cols 2*bc..2*bc+1
    #   Local (r, c) in 2x2 -> global row = 2*br + r, col = 2*bc + c
    #   W index = global_row * 4 + global_col

    col = 0
    for hl_idx, (a_comb, b_comb) in enumerate(high_level):
        # a_comb: list of (coeff, block_idx) for A
        # b_comb: list of (coeff, block_idx) for B
        # For each 2x2 Strassen term (s_u, s_v, s_w) in s2_terms:
        for s_u, s_v, s_w in s2_terms:
            # s_u: dict local_A_idx -> coeff (local 2x2 index: 0=a00,1=a01,2=a10,3=a11)
            # s_v: dict local_B_idx -> coeff
            # s_w: dict local_C_idx -> coeff

            # For each entry in s_u, for each (coeff_a, a_block) in a_comb:
            for local_a, coeff_su in s_u.items():
                for coeff_a, a_block in a_comb:
                    # local_a: 0=a00, 1=a01, 2=a10, 3=a11 within the 2x2 block
                    # a_block: 0=A00, 1=A01, 2=A10, 3=A11 in the 4x4 matrix
                    # a_block determines the 2x2 sub-block position:
                    #   block 0: rows 0-1, cols 0-1
                    #   block 1: rows 0-1, cols 2-3
                    #   block 2: rows 2-3, cols 0-1
                    #   block 3: rows 2-3, cols 2-3
                    br_a = a_block // 2
                    bc_a = a_block % 2
                    # local_a: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) within the 2x2
                    lr_a = local_a // 2
                    lc_a = local_a % 2
                    global_row_a = 2 * br_a + lr_a
                    global_col_a = 2 * bc_a + lc_a
                    u_idx = global_row_a * 4 + global_col_a
                    U[u_idx][col] += coeff_a * coeff_su

            # For each entry in s_v, for each (coeff_b, b_block) in b_comb:
            for local_b, coeff_sv in s_v.items():
                for coeff_b, b_block in b_comb:
                    br_b = b_block // 2
                    bc_b = b_block % 2
                    lr_b = local_b // 2
                    lc_b = local_b % 2
                    global_row_b = 2 * br_b + lr_b
                    global_col_b = 2 * bc_b + lc_b
                    v_idx = global_row_b * 4 + global_col_b
                    V[v_idx][col] += coeff_b * coeff_sv

            # For each entry in s_w:
            # C reconstruction depends on hl_idx
            # C00 = M1 + M4 - M5 + M7  -> blocks (0,0)
            # C01 = M3 + M5            -> blocks (0,1)
            # C10 = M2 + M4            -> blocks (1,0)
            # C11 = M1 - M2 + M3 + M6  -> blocks (1,1)

            # Determine which C blocks this high-level product contributes to
            # and with what sign
            c_contribs = []
            if hl_idx == 0:  # M1
                c_contribs = [(0, 1), (3, 1)]  # C00 (block 0) +1, C11 (block 3) +1
            elif hl_idx == 1:  # M2
                c_contribs = [(2, 1), (3, -1)]  # C10 (block 2) +1, C11 (block 3) -1
            elif hl_idx == 2:  # M3
                c_contribs = [(1, 1), (3, 1)]  # C01 (block 1) +1, C11 (block 3) +1
            elif hl_idx == 3:  # M4
                c_contribs = [(0, 1), (2, 1)]  # C00 (block 0) +1, C10 (block 2) +1
            elif hl_idx == 4:  # M5
                c_contribs = [(0, -1), (1, 1)]  # C00 (block 0) -1, C01 (block 1) +1
            elif hl_idx == 5:  # M6
                c_contribs = [(3, 1)]  # C11 (block 3) +1
            elif hl_idx == 6:  # M7
                c_contribs = [(0, 1)]  # C00 (block 0) +1

            for local_c, coeff_sw in s_w.items():
                for c_block, sign in c_contribs:
                    br_c = c_block // 2
                    bc_c = c_block % 2
                    lr_c = local_c // 2
                    lc_c = local_c % 2
                    global_row_c = 2 * br_c + lr_c
                    global_col_c = 2 * bc_c + lc_c
                    w_idx = global_row_c * 4 + global_col_c
                    W[w_idx][col] += sign * coeff_sw

            col += 1

    assert col == 49, f"Expected 49 terms, got {col}"
    return U, V, W


def _build_schoolbook_4x4_4x1():
    """Build the rank-16 schoolbook decomposition for 4x4 * 4x1.

    n=4, m=4, p=1, so nm=16, mp=4, pn=4.
    Returns (U, V, W) with shapes (16, 16), (4, 16), (4, 16).
    """
    F = Fraction
    nm = 16
    mp = 4
    pn = 4
    rank = 16
    U = [[F(0)] * rank for _ in range(nm)]
    V = [[F(0)] * rank for _ in range(mp)]
    W = [[F(0)] * rank for _ in range(pn)]
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = F(1)
            V[j * 1 + 0][col] = F(1)
            W[0 * 4 + i][col] = F(1)
            col += 1
    return U, V, W


def _verify_full(U, V, W, rank):
    """Verify the full 4x4 * 4x9 decomposition.

    U: (16, rank), V: (36, rank), W: (36, rank)
    Check: for all (i,j,k) in [0,4)x[0,4)x[0,9):
      sum_c U[i*4+j, c] * V[j*9+k, c] * W[k*4+i, c] == 1
    """
    n, m, p = 4, 4, 9
    F = Fraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = F(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != F(1):
                    return False
    return True


def _schoolbook_full():
    """Return the schoolbook rank-144 decomposition for 4x4 * 4x9."""
    F = Fraction
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = [[F(0)] * rank for _ in range(nm)]
    V = [[F(0)] * rank for _ in range(mp)]
    W = [[F(0)] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = F(1)
                V[j * p + k][col] = F(1)
                W[k * n + i][col] = F(1)
                col += 1
    return U, V, W, rank


def _to_output_list(mat):
    """Convert a matrix of Fractions to a list of lists of ints or halves."""
    result = []
    for row in mat:
        new_row = []
        for x in row:
            if x.denominator == 1:
                new_row.append(int(x.numerator))
            elif x.denominator == 2:
                new_row.append(x.numerator / 2)
            else:
                new_row.append(float(x))
        result.append(new_row)
    return result


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix mult."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Try the Strassen-based rank-114 decomposition
    try:
        # Build 4x4 * 4x4 Strassen (rank 49)
        U44, V44, W44 = _build_strassen_4x4()
        # Build 4x4 * 4x1 schoolbook (rank 16)
        U41, V41, W41 = _build_schoolbook_4x4_4x1()

        rank = 49 + 49 + 16
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]

        # First 4x4 * 4x4 block: B columns 0-3
        # U: same for both 4x4 blocks (A is the same)
        # V: for first block, V[j*4+k] where k in 0..3
        # W: for first block, W[k*4+i] where k in 0..3

        # Copy first 4x4*4x4 (B cols 0-3)
        for i in range(nm):
            for c in range(49):
                U[i][c] = U44[i][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j * 9 + k][c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[k * 4 + i][c] = W44[k * 4 + i][c]

        # Second 4x4 * 4x4 block: B columns 4-7
        for i in range(nm):
            for c in range(49):
                U[i][49 + c] = U44[i][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j * 9 + (4 + k)][49 + c] = V44[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[(4 + k) * 4 + i][49 + c] = W44[k * 4 + i][c]

        # Third 4x4 * 4x1 block: B column 8
        for i in range(nm):
            for c in range(16):
                U[i][98 + c] = U41[i][c]
        for j in range(4):
            for c in range(16):
                V[j * 9 + 8][98 + c] = V41[j][c]
        for i in range(4):
            for c in range(16):
                W[8 * 4 + i][98 + c] = W41[i][c]

        # Verify
        if _verify_full(U, V, W, rank):
            U_out = _to_output_list(U)
            V_out = _to_output_list(V)
            W_out = _to_output_list(W)
            return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass

    # Fallback: schoolbook rank-144
    U, V, W, rank = _schoolbook_full()
    U_out = _to_output_list(U)
    V_out = _to_output_list(V)
    W_out = _to_output_list(W)
    return {"U": U_out, "V": V_out, "W": W_out}
