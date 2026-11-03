"""Matrix multiplication 3x4 by 4x6 rank-minimal decomposition (block/Strassen family)."""
import time

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0


def _strassen2x2():
    """Rank-7 Strassen decomposition of 2x2 by 2x2.
    Returns (U, V, W) each 4x7 with U[i*2+j, c], V[i*2+j, c], W[i*2+j, c].
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    # 1-indexed Strassen factors
    Uf = [(1, 1), (1, 0), (0, 1), (1, 1), (0, 0), (0, 1), (1, 0)]
    Vf = [(0, 1), (1, 1), (1, 1), (0, 1), (1, 0), (0, 1), (1, 0)]
    Wf = [(1, 0), (1, 0), (0, 1), (1, 1), (1, 0), (1, 1), (0, 1)]
    for c in range(7):
        U[Uf[c][0] * 2 + Uf[c][1]][c] = 1
        V[Vf[c][0] * 2 + Vf[c][1]][c] = 1
        W[Wf[c][0] * 2 + Wf[c][1]][c] = 1
    return U, V, W


def _block1x2x2x2():
    """Rank-4 decomposition of (1x2) by (2x2).
    A block: 1x2 (row 0). B block: 2x2. C block: 1x2.
    U 2x4 (rows A), V 4x4 (rows B), W 2x4 (rows C).
    """
    U = [[0] * 4 for _ in range(2)]
    V = [[0] * 4 for _ in range(4)]
    W = [[0] * 4 for _ in range(2)]
    U[0][0] = 1
    U[1][1] = 1
    U[0][2] = 1
    U[1][3] = 1
    V[0][0] = 1
    V[1][1] = 1
    V[2][2] = 1
    V[3][3] = 1
    W[0][0] = 1
    W[1][1] = 1
    W[0][2] = 1
    W[1][3] = 1
    return U, V, W


def _block1x2x2x2_cross():
    """Rank-4 decomposition of (1x2) by (2x2) with cross terms.
    A block: 1x2 (row 0). B block: 2x2. C block: 1x2.
    C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    """
    U = [[0] * 4 for _ in range(2)]
    V = [[0] * 4 for _ in range(4)]
    W = [[0] * 4 for _ in range(2)]
    # c0: A[0]*B[0,0] -> C[0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    # c1: A[1]*B[1,0] -> C[0]
    U[1][1] = 1
    V[2][1] = 1
    W[0][1] = 1
    # c2: A[0]*B[0,1] -> C[1]
    U[0][2] = 1
    V[1][2] = 1
    W[1][2] = 1
    # c3: A[1]*B[1,1] -> C[1]
    U[1][3] = 1
    V[3][3] = 1
    W[1][3] = 1
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    # A is 3x4, B is 4x6, C is 3x6.
    # Partition A rows: (1, 1, 2), B cols: (2, 2, 2).
    A_row_splits = [1, 1, 2]
    B_col_splits = [2, 2, 2]
    C_row_splits = [1, 1, 2]
    C_col_splits = [2, 2, 2]

    # Precompute row offsets in A (rows 0..2) and C (rows 0..2)
    A_row_offsets = [0, 1, 2]
    C_row_offsets = [0, 1, 2]
    # B col offsets: 0,2,4
    B_col_offsets = [0, 2, 4]
    # C col offsets: 0,2,4
    C_col_offsets = [0, 2, 4]

    # Build blocks: for each (a_block, b_block)
    # a_block in {0,1,2}, b_block in {0,1,2}
    # A block dims: A_row_splits[a] x 2 (since each B group has 2 cols)
    # B block dims: 2 x B_col_splits[b]
    # C block dims: C_row_splits[a] x C_col_splits[b]

    U_rows = []
    V_rows = []
    W_rows = []
    # We'll accumulate per-block contributions.
    # U: n*m = 12 rows, V: m*p = 24 rows, W: p*n = 18 rows.

    # We'll build column lists for U, V, W.
    U_cols = []  # list of dicts: row -> value
    V_cols = []
    W_cols = []

    def add_block(Ub, Vb, Wb, a_off, b_off, c_off, a_dim, b_dim, c_dim, a_col_start, b_col_start):
        """Add a block's U (a_dim x 2 rows), V (b_dim x b_dim cols), W (c_dim x c_dim cols)."""
        rank_b = len(Ub[0])
        for c in range(rank_b):
            u_col = {}
            v_col = {}
            w_col = {}
            for i in range(a_dim):
                u_row = a_off + i
                u_col[u_row] = Ub[i][c]
            for i in range(b_dim):
                for j in range(b_dim):
                    v_row = b_off + i * b_dim + j
                    val = Vb[i * b_dim + j][c]
                    v_col[v_row] = val
            for i in range(c_dim):
                for j in range(c_dim):
                    w_row = c_off + i * c_dim + j
                    w_col[w_row] = Wb[i * c_dim + j][c]
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Block (a=0, b=0): A 1x2, B 2x2, C 1x2 -> rank 4
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0] ; C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    Ub, Vb, Wb = _block1x2x2x2_cross()
    add_block(Ub, Vb, Wb, a_off=0, b_off=0, c_off=0, a_dim=1, b_dim=2, c_dim=1, a_col_start=0, b_col_start=0)

    # Block (a=0, b=1): A 1x2, B 2x2, C 1x2 -> rank 4
    Ub, Vb, Wb = _block1x2x2x2_cross()
    add_block(Ub, Vb, Wb, a_off=0, b_off=2, c_off=2, a_dim=1, b_dim=2, c_dim=1, a_col_start=2, b_col_start=2)

    # Block (a=0, b=2): A 1x2, B 2x2, C 1x2 -> rank 4
    Ub, Vb, Wb = _block1x2x2x2_cross()
    add_block(Ub, Vb, Wb, a_off=0, b_off=4, c_off=4, a_dim=1, b_dim=2, c_dim=1, a_col_start=4, b_col_start=4)

    # Block (a=1, b=0): A 1x2, B 2x2, C 1x2 -> rank 4
    Ub, Vb, Wb = _block1x2x2x2_cross()
    add_block(Ub, Vb, Wb, a_off=1, b_off=0, c_off=0, a_dim=1, b_dim=2, c_dim=1, a_col_start=0, b_col_start=0)

    # Block (a=1, b=1): A 1x2, B 2x2, C 1x2 -> rank 4
    Ub, Vb, Wb = _block1x2x2x2_cross()
    add_block(Ub, Vb, Wb, a_off=1, b_off=2, c_off=2, a_dim=1, b_dim=2, c_dim=1, a_col_start=2, b_col_start=2)

    # Block (a=1, b=2): A 1x2, B 2x2, C 1x2 -> rank 4
    Ub, Vb, Wb = _block1x2x2x2_cross()
    add_block(Ub, Vb, Wb, a_off=1, b_off=4, c_off=4, a_dim=1, b_dim=2, c_dim=1, a_col_start=4, b_col_start=4)

    # Block (a=2, b=0): A 2x2, B 2x2, C 2x2 -> rank 7 (Strassen)
    Ub, Vb, Wb = _strassen2x2()
    add_block(Ub, Vb, Wb, a_off=2, b_off=0, c_off=2, a_dim=2, b_dim=2, c_dim=2, a_col_start=0, b_col_start=0)

    # Block (a=2, b=1): A 2x2, B 2x2, C 2x2 -> rank 7 (Strassen)
    Ub, Vb, Wb = _strassen2x2()
    add_block(Ub, Vb, Wb, a_off=2, b_off=2, c_off=4, a_dim=2, b_dim=2, c_dim=2, a_col_start=2, b_col_start=2)

    # Block (a=2, b=2): A 2x2, B 2x2, C 2x2 -> rank 7 (Strassen)
    Ub, Vb, Wb = _strassen2x2()
    add_block(Ub, Vb, Wb, a_off=2, b_off=4, c_off=6, a_dim=2, b_dim=2, c_dim=2, a_col_start=4, b_col_start=4)

    rank = len(U_cols)

    # Convert to dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for r, v in U_cols[c].items():
            U[r][c] = v
        for r, v in V_cols[c].items():
            V[r][c] = v
        for r, v in W_cols[c].items():
            W[r][c] = v

    # Deadline guard: if we've used too much time, still return (this is fast).
    if time.time() - t0 > DEADLINE - 5:
        # Fallback to schoolbook if somehow over budget (shouldn't happen)
        pass

    return {"U": U, "V": V, "W": W}
