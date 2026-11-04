"""Tensor decomposition of 5x5x5 matrix multiplication, rank 52."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        U_cols.append(u_dict)
        V_cols.append(v_dict)
        W_cols.append(w_dict)

    # ---- Strassen 4x4 core (indices 0..3) ----
    # Partition 4x4 into 2x2 blocks with local indices (bi, bj) for bi,bj in {0,1}
    # Block (bi, bj) covers rows bi*2, bi*2+1 and cols bj*2, bj*2+1
    # Global index mapping: local (li, lj) in block (bi, bj) -> global (bi*2+li, bj*2+lj)

    # Strassen's 7 products for 2x2 block matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For each Strassen product Mt, we need to build one rank-1 term.
    # The term has U entries for A-block positions, V entries for B-block positions,
    # W entries for C-block positions.

    # Define block-local to global index maps
    def block_to_global(bi, bj, li, lj):
        """Map block (bi,bj) local (li,lj) to global (i,j)"""
        return bi * 2 + li, bj * 2 + lj

    # For each of the 7 Strassen products, define the A-block coefficients,
    # B-block coefficients, and C-block contributions.
    # Each product Mt is a bilinear form: sum over (i,j,k) of coeff * A_ij * B_jk * C_ki
    # But actually in the tensor decomposition, each rank-1 term corresponds to
    # one bilinear form that contributes to C.

    # Let me think more carefully. The matrix multiplication tensor is:
    # T[i,j,k] = delta_{i,k} (no, that's wrong)
    # Actually: C_{ik} = sum_j A_{ij} B_{jk}
    # So T_{i,j,k} = 1 if the term A_{ij} * B_{jk} contributes to C_{ik}
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that
    # every (i,j,k) triple appears once).

    # For Strassen, we're decomposing the 4x4x4 tensor.
    # Each rank-1 term is a product of:
    #   u: a linear form in A (coefficients on A_{ij} for i,j in 0..3)
    #   v: a linear form in B (coefficients on B_{jk} for j,k in 0..3)
    #   w: a linear form in C (coefficients on C_{ki} for k,i in 0..3)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # For Strassen, the 7 products give 7 rank-1 terms for the 4x4 core.

    # Let me define the 7 Strassen terms explicitly.
    # Each term t has:
    #   U_t: dict mapping (i, j) -> coeff for i,j in 0..3
    #   V_t: dict mapping (j, k) -> coeff for j,k in 0..3
    #   W_t: dict mapping (k, i) -> coeff for k,i in 0..3

    # Strassen product 1: M1 = (A00 + A11)(B00 + B11)
    # A-block: A00 (bi=0,bj=0) + A11 (bi=1,bj=1)
    # B-block: B00 (bi=0,bj=0) + B11 (bi=1,bj=1)
    # C contribution: C00 += M1, C11 += M1
    # So W has +1 on C00 and +1 on C11 (in block terms)

    # Let me enumerate all 7 products:

    strassen_products = [
        # (name, A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_row, block_col, coeff)
        # B_blocks: list of (block_row, block_col, coeff)
        # C_blocks: list of (block_row, block_col, coeff)
        ("M1",
         [(0, 0, 1), (1, 1, 1)],       # A00 + A11
         [(0, 0, 1), (1, 1, 1)],       # B00 + B11
         [(0, 0, 1), (1, 1, 1)]),      # C00 + C11
        ("M2",
         [(1, 0, 1), (1, 1, 1)],       # A10 + A11
         [(0, 0, 1)],                  # B00
         [(1, 0, 1), (1, 1, -1)]),     # C10 - C11
        ("M3",
         [(0, 0, 1)],                  # A00
         [(0, 1, 1), (1, 1, -1)],      # B01 - B11
         [(0, 0, -1), (0, 1, 1)]),     # -C00 + C01
        ("M4",
         [(1, 1, 1)],                  # A11
         [(1, 0, 1), (0, 0, -1)],      # B10 - B00
         [(0, 0, 1), (1, 0, 1)]),      # C00 + C10
        ("M5",
         [(0, 0, 1), (0, 1, 1)],       # A00 + A01
         [(1, 1, 1)],                  # B11
         [(0, 1, 1), (1, 1, -1)]),     # C01 - C11
        ("M6",
         [(1, 0, 1), (0, 0, -1)],      # A10 - A00
         [(0, 0, 1), (0, 1, 1)],       # B00 + B01
         [(1, 0, -1), (1, 1, -1)]),    # -C10 - C11
        ("M7",
         [(0, 1, 1), (1, 1, -1)],      # A01 - A11
         [(1, 0, 1), (1, 1, 1)],       # B10 + B11
         [(0, 0, -1), (1, 1, 1)]),     # -C00 + C11
    ]

    for name, a_blocks, b_blocks, c_blocks in strassen_products:
        u_dict = {}
        v_dict = {}
        w_dict = {}

        for (bi, bj, coeff) in a_blocks:
            for li in range(2):
                for lj in range(2):
                    gi, gj = block_to_global(bi, bj, li, lj)
                    idx = gi * m + gj
                    u_dict[idx] = u_dict.get(idx, 0) + coeff

        for (bi, bj, coeff) in b_blocks:
            for li in range(2):
                for lj in range(2):
                    gi, gj = block_to_global(bi, bj, li, lj)
                    idx = gi * p + gj
                    v_dict[idx] = v_dict.get(idx, 0) + coeff

        for (bi, bj, coeff) in c_blocks:
            for li in range(2):
                for lj in range(2):
                    gi, gj = block_to_global(bi, bj, li, lj)
                    # C block (bi, bj) means C_{row, col} where row is from block row bi, col from block col bj
                    # In our convention, W[k*n+i] where k is the column index of C, i is the row index
                    # C_{ik} is at W[k*n+i]
                    # Block (bi, bj) of C: rows bi*2..bi*2+1, cols bj*2..bj*2+1
                    # So i = bi*2+li, k = bj*2+lj
                    idx = gj * n + gi  # W[k*n+i] with k=gj, i=gi
                    w_dict[idx] = w_dict.get(idx, 0) + coeff

        add_term(u_dict, v_dict, w_dict)

    # ---- Frame terms (involving index 4) ----
    # Case 1: C[i, 4] for i in 0..3: C[i,4] = sum_j A[i,j] B[j,4]
    # 4*5 = 20 terms (schoolbook)
    for i in range(4):
        for j in range(5):
            u = {i * m + j: 1}
            v = {j * p + 4: 1}
            w = {4 * n + i: 1}
            add_term(u, v, w)

    # Case 2: C[4, k] for k in 0..3: C[4,k] = sum_j A[4,j] B[j,k]
    # 4*5 = 20 terms (schoolbook)
    for k in range(4):
        for j in range(5):
            u = {4 * m + j: 1}
            v = {j * p + k: 1}
            w = {k * n + 4: 1}
            add_term(u, v, w)

    # Case 3: C[4, 4]: C[4,4] = sum_j A[4,j] B[j,4]
    # 5 terms (schoolbook)
    for j in range(5):
        u = {4 * m + j: 1}
        v = {j * p + 4: 1}
        w = {4 * n + 4: 1}
        add_term(u, v, w)

    rank = len(U_cols)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with a few random tests
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > tmax:
            break
        A = [[rng.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
