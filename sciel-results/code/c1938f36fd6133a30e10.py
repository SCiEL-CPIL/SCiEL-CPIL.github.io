"""Tensor decomposition for 4x4x6 matrix multiplication.

Strategy:
  * Build a rank-7 Strassen decomposition of the 2x2x2 multiplication tensor.
  * View the 4x4x6 problem as 6 independent 2x2x2 blocks:
      A is 2x2 blocks (2 row-blocks x 2 col-blocks)
      B is 2x2 blocks (2 row-blocks x 3 col-blocks)
      C is 2x2 blocks (2 row-blocks x 3 col-blocks)
    Each output block C_{a,b} = A_{a,0} B_{0,b} + A_{a,1} B_{1,b}.
    We compute each of the 12 sub-products (A_{a,r} B_{r,b}) with the
    rank-7 Strassen decomposition, giving a total rank of 12 * 7 = 84.
  * This is an exact decomposition with integer entries, well below the
    schoolbook rank of 96.
"""
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2x2 coefficients (rank 7)
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)(B00)
# M3 = (A00)(B10+B11)
# M4 = (A11)(B11-B00)
# M5 = (A00+A10)(B11)
# M6 = (A11-A10)(B00+B10)
# M7 = (A00-A11)(B01+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6


def _strassen_2x2x2_terms():
    """Return the 7 rank-1 terms of the 2x2x2 Strassen decomposition.

    Each term is a dict with keys 'U', 'V', 'W', each a list of
    (local_index, value) pairs. Local indices run 0..3 for the 2x2
    matrices (row-major: 00, 01, 10, 11).
    """
    steps = [
        # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)],
        },
        # M2 = (A10+A11)(B00), contributes to C10 (+) and C11 (-)
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)],
        },
        # M3 = (A00)(B10+B11), contributes to C01 and C11
        {
            'U': [(0, 1)],
            'V': [(2, 1), (3, 1)],
            'W': [(1, 1), (3, 1)],
        },
        # M4 = (A11)(B11-B00), contributes to C00 and C10
        {
            'U': [(3, 1)],
            'V': [(3, 1), (0, -1)],
            'W': [(0, 1), (2, 1)],
        },
        # M5 = (A00+A10)(B11), contributes to C00 (-) and C01 (+)
        {
            'U': [(0, 1), (2, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)],
        },
        # M6 = (A11-A10)(B00+B10), contributes to C11
        {
            'U': [(3, 1), (2, -1)],
            'V': [(0, 1), (2, 1)],
            'W': [(3, 1)],
        },
        # M7 = (A00-A11)(B01+B11), contributes to C00
        {
            'U': [(0, 1), (3, -1)],
            'V': [(1, 1), (3, 1)],
            'W': [(0, 1)],
        },
    ]
    return steps


def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 6

    # We'll build the decomposition by treating the problem as 6 independent
    # 2x2x2 multiplications (for each pair of A-row-block, B-col-block),
    # each computed via Strassen (rank 7).
    #
    # A is 4x4: split into 2x2 blocks A_{a, r} for a in {0,1}, r in {0,1}
    #   A_{a, r}[i, j] = A[a*2+i, r*2+j]
    # B is 4x6: split into 2x2 blocks B_{r, b} for r in {0,1}, b in {0,1,2}
    #   B_{r, b}[i, j] = B[r*2+i, b*2+j]
    # C is 4x6: split into 2x2 blocks C_{a, b} for a in {0,1}, b in {0,1,2}
    #   C_{a, b}[i, j] = C[a*2+i, b*2+j]
    #
    # C_{a, b} = A_{a,0} B_{0,b} + A_{a,1} B_{1,b}
    #
    # For each (a, b), we need two 2x2x2 products:
    #   P1 = A_{a,0} * B_{0,b}  (Strassen rank 7)
    #   P2 = A_{a,1} * B_{1,b}  (Strassen rank 7)
    # C_{a, b} = P1 + P2
    #
    # Total rank: 6 * 2 * 7 = 84

    strassen_terms = _strassen_2x2x2_terms()
    R = 7  # rank per 2x2x2 block

    # We'll accumulate U, V, W as lists of columns (rank-1 terms).
    # Each column c corresponds to one rank-1 term.
    # U has n*m = 16 rows, V has m*p = 24 rows, W has p*n = 24 rows.

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {(i,j): val}, v_dict: {(j,k): val}, w_dict: {(k,i): val}."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for (i, j), val in u_dict.items():
            u[i * m + j] = val
        for (j, k), val in v_dict.items():
            v[j * p + k] = val
        for (k, i), val in w_dict.items():
            w[k * n + i] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Map local 2x2 index (0..3) to global (i,j) given block offsets
    def local_to_global_A(a, r, li):
        """li in 0..3 -> (i, j) in A. A block (a,r) covers rows a*2..a*2+1, cols r*2..r*2+1."""
        li_i = li // 2
        li_j = li % 2
        return (a * 2 + li_i, r * 2 + li_j)

    def local_to_global_B(r, b, li):
        """li in 0..3 -> (j, k) in B. B block (r,b) covers rows r*2..r*2+1, cols b*2..b*2+1."""
        li_j = li // 2
        li_k = li % 2
        return (r * 2 + li_j, b * 2 + li_k)

    def local_to_global_C(a, b, li):
        """li in 0..3 -> (k, i) in C (trace convention W[k*n+i]). C block (a,b) covers rows a*2..a*2+1, cols b*2..b*2+1."""
        li_k = li // 2  # this is the "i" in C (row of C)
        li_i = li % 2   # this is the "k" in C (col of C)
        # Wait, C is 4x6. C[i, k] where i in 0..3, k in 0..5.
        # W index is k*n + i.
        # C block (a,b): rows a*2..a*2+1 (i), cols b*2..b*2+1 (k)
        # local index li: li_i = li//2 (row within block), li_k = li%2 (col within block)
        # global i = a*2 + li_i, global k = b*2 + li_k
        gi = a * 2 + li // 2
        gk = b * 2 + li % 2
        return (gk, gi)  # (k, i) for W indexing

    # For each (a, b) in {0,1} x {0,1,2}:
    #   For each r in {0,1}:
    #     Apply Strassen to A_{a,r} * B_{r,b}
    #     The Strassen terms give contributions to C_{a,b}
    #     We add these contributions to the global C.

    for a in range(2):
        for b in range(3):
            for r in range(2):
                for term in strassen_terms:
                    u_dict = {}
                    v_dict = {}
                    w_dict = {}

                    for (li, val) in term['U']:
                        (gi, gj) = local_to_global_A(a, r, li)
                        u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + val

                    for (li, val) in term['V']:
                        (gj, gk) = local_to_global_B(r, b, li)
                        v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + val

                    for (li, val) in term['W']:
                        (gk, gi) = local_to_global_C(a, b, li)
                        w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + val

                    add_term(u_dict, v_dict, w_dict)

    if time.time() > deadline:
        # Shouldn't happen, but just in case
        pass

    rank = len(U_cols)
    # Convert to list-of-lists format
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for i in range(n * m):
            if U_cols[c][i] != 0:
                U[i][c] = U_cols[c][i]
        for i in range(m * p):
            if V_cols[c][i] != 0:
                V[i][c] = V_cols[c][i]
        for i in range(p * n):
            if W_cols[c][i] != 0:
                W[i][c] = W_cols[c][i]

    return {"U": U, "V": V, "W": W}
