"""Tensor decomposition for 3x4x5 matrix multiplication using block-structured Strassen."""
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=4, p=5):
    """Verify the decomposition exactly using multiple random tests."""
    rank = len(U[0]) if U else 0
    for seed in [12345, 67890, 11111, 22222, 33333]:
        rng = random.Random(seed)
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C[i][k] = s
        Cd = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                Cd[i][k] = s
        for i in range(n):
            for k in range(p):
                if Cd[i][k] != C[i][k]:
                    return False
    return True


def _build_strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term: (u_entries, v_entries, w_entries)
    where u_entries: dict mapping (i,j) -> coef for U
    v_entries: dict mapping (j,k) -> coef for V
    w_entries: dict mapping (k,i) -> coef for W
    """
    terms = []
    # M1 = (A00+A11)(B01-B11) -> C11
    terms.append({
        'u': {(0,0): 1, (1,1): 1},
        'v': {(0,1): 1, (1,1): -1},
        'w': {(1,1): 1}
    })
    # M2 = A10(B00-B11) -> C10
    terms.append({
        'u': {(1,0): 1},
        'v': {(0,0): 1, (1,1): -1},
        'w': {(0,1): 1}
    })
    # M3 = A01(B10+B11) -> C01
    terms.append({
        'u': {(0,1): 1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(1,0): 1}
    })
    # M4 = (A00-A10)(B10+B11) -> C00
    terms.append({
        'u': {(0,0): 1, (1,0): -1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(0,0): 1}
    })
    # M5 = (A01+A11)(B00-B10) -> C01
    terms.append({
        'u': {(0,1): 1, (1,1): 1},
        'v': {(0,0): 1, (1,0): -1},
        'w': {(0,1): 1}
    })
    # M6 = (A00-A11)(B01+B11) -> C00
    terms.append({
        'u': {(0,0): 1, (1,1): -1},
        'v': {(0,1): 1, (1,1): 1},
        'w': {(1,0): 1}
    })
    # M7 = (A10-A00)(B00+A11) -> C10
    terms.append({
        'u': {(1,0): 1, (0,0): -1},
        'v': {(0,0): 1, (1,1): 1},
        'w': {(0,1): 1}
    })
    return terms


def _build_decomposition():
    """
    Build a block-structured decomposition for 3x4x5.
    
    Strategy:
    - Split A (3x4) into A_top (2x4) and A_bot (1x4)
    - Split B (4x5) into B_left (4x3) and B_right (4x2)
    - C_top (2x5) = A_top * B
    - C_bot (1x5) = A_bot * B
    
    For C_top (2x4x5):
      Split A_top into A_tl (2x2), A_tr (2x2)
      Split B into B_top (2x5), B_bot (2x5)
      C_top = A_tl*B_top + A_tr*B_bot
      Each 2x2x5 block: split into 2x2x2 + 2x2x2 + 2x2x1
      2x2x2 -> Strassen (rank 7)
      2x2x1 -> schoolbook (rank 2)
      So each 2x2x5 -> rank 7+7+2 = 16
      Total for C_top: 2*16 = 32
    
    For C_bot (1x4x5):
      Split A_bot into A_bl (1x2), A_br (1x2)
      C_bot = A_bl*B_top + A_br*B_bot
      Each 1x2x5 block: split into 1x2x2 + 1x2x2 + 1x2x1
      1x2x2 -> rank 2 (just two dot products)
      1x2x1 -> rank 1
      So each 1x2x5 -> rank 2+2+1 = 5
      Total for C_bot: 2*5 = 10
    
    Total rank: 32 + 10 = 42
    """
    n, m, p = 3, 4, 5
    
    # We'll build U, V, W as dictionaries first, then convert to lists
    # U: (n*m, rank) -> index by (i, j) where i in [0,n), j in [0,m)
    # V: (m*p, rank) -> index by (j, k) where j in [0,m), k in [0,p)
    # W: (p*n, rank) -> index by (k, i) where k in [0,p), i in [0,n)
    
    U_dict = {}  # (i,j) -> list of (col, coef)
    V_dict = {}  # (j,k) -> list of (col, coef)
    W_dict = {}  # (k,i) -> list of (col, coef)
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries: dict {(i,j): coef}, etc."""
        nonlocal col
        for (i, j), c in u_entries.items():
            if c != 0:
                U_dict.setdefault((i, j), []).append((col, c))
        for (j, k), c in v_entries.items():
            if c != 0:
                V_dict.setdefault((j, k), []).append((col, c))
        for (k, i), c in w_entries.items():
            if c != 0:
                W_dict.setdefault((k, i), []).append((col, c))
        col += 1
    
    # Helper: add Strassen 2x2x2 with offset
    def add_strassen(a_row_offset, a_col_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen 2x2x2 decomposition for:
        A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2] *
        B[a_col_offset:a_col_offset+2, b_col_offset:b_col_offset+2] ->
        C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        """
        strassen_terms = _build_strassen_2x2x2()
        for term in strassen_terms:
            u_entries = {}
            v_entries = {}
            w_entries = {}
            for (di, dj), c in term['u'].items():
                i = a_row_offset + di
                j = a_col_offset + dj
                u_entries[(i, j)] = c
            for (dj, dk), c in term['v'].items():
                j = a_col_offset + dj
                k = b_col_offset + dk
                v_entries[(j, k)] = c
            for (dk, di), c in term['w'].items():
                k = b_col_offset + dk
                i = c_row_offset + di
                w_entries[(k, i)] = c
            add_term(u_entries, v_entries, w_entries)
    
    # Helper: add schoolbook 2x2x1
    def add_2x2x1(a_row_offset, a_col_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        2x2 * 2x1 -> 2x1
        A[2x2] * B[2x1] -> C[2x1]
        C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        Rank 2.
        """
        # Term 1: A[0,0]*B[0] -> C[0], A[1,0]*B[0] -> C[1]
        add_term(
            {(a_row_offset, a_col_offset): 1, (a_row_offset+1, a_col_offset): 1},
            {(a_col_offset, b_col_offset): 1},
            {(b_col_offset, c_row_offset): 1, (b_col_offset, c_row_offset+1): 1}
        )
        # Term 2: A[0,1]*B[1] -> C[0], A[1,1]*B[1] -> C[1]
        add_term(
            {(a_row_offset, a_col_offset+1): 1, (a_row_offset+1, a_col_offset+1): 1},
            {(a_col_offset+1, b_col_offset): 1},
            {(b_col_offset, c_row_offset): 1, (b_col_offset, c_row_offset+1): 1}
        )
    
    # Helper: add schoolbook 1x2x2
    def add_1x2x2(a_row_offset, a_col_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        1x2 * 2x2 -> 1x2
        A[1x2] * B[2x2] -> C[1x2]
        C[0] = A[0]*B[0,0] + A[1]*B[1,0]
        C[1] = A[0]*B[0,1] + A[1]*B[1,1]
        Rank 2.
        """
        # Term 1: A[0]*B[0,0] -> C[0], A[0]*B[0,1] -> C[1]
        add_term(
            {(a_row_offset, a_col_offset): 1},
            {(a_col_offset, b_col_offset): 1, (a_col_offset, b_col_offset+1): 1},
            {(b_col_offset, c_row_offset): 1, (b_col_offset+1, c_row_offset): 1}
        )
        # Term 2: A[1]*B[1,0] -> C[0], A[1]*B[1,1] -> C[1]
        add_term(
            {(a_row_offset, a_col_offset+1): 1},
            {(a_col_offset+1, b_col_offset): 1, (a_col_offset+1, b_col_offset+1): 1},
            {(b_col_offset, c_row_offset): 1, (b_col_offset+1, c_row_offset): 1}
        )
    
    # Helper: add schoolbook 1x2x1
    def add_1x2x1(a_row_offset, a_col_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        1x2 * 2x1 -> 1x1
        A[1x2] * B[2x1] -> C[1x1]
        C[0] = A[0]*B[0] + A[1]*B[1]
        Rank 2.
        """
        add_term(
            {(a_row_offset, a_col_offset): 1},
            {(a_col_offset, b_col_offset): 1},
            {(b_col_offset, c_row_offset): 1}
        )
        add_term(
            {(a_row_offset, a_col_offset+1): 1},
            {(a_col_offset+1, b_col_offset): 1},
            {(b_col_offset, c_row_offset): 1}
        )
    
    # ===== C_top (2x4x5): rows 0,1 of A =====
    # Split A_top (2x4) into A_tl (2x2, cols 0-1), A_tr (2x2, cols 2-3)
    # Split B (4x5) into B_top (2x5, rows 0-1), B_bot (2x5, rows 2-3)
    
    # C_top = A_tl * B_top + A_tr * B_bot
    # Each product is 2x2x5
    
    # A_tl * B_top: A[0:2, 0:2] * B[0:2, 0:5] -> C[0:2, 0:5]
    # Split B_top into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    # 2x2x2 + 2x2x2 + 2x2x1
    add_strassen(0, 0, 0, 0, 0)      # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 0, 2, 0, 2)      # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_2x2x1(0, 0, 4, 0, 4)         # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    
    # A_tr * B_bot: A[0:2, 2:4] * B[2:4, 0:5] -> C[0:2, 0:5]
    # Split B_bot into B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    add_strassen(0, 2, 0, 0, 0)      # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0, 2)      # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    add_2x2x1(0, 2, 4, 0, 4)         # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    
    # ===== C_bot (1x4x5): row 2 of A =====
    # Split A_bot (1x4) into A_bl (1x2, cols 0-1), A_br (1x2, cols 2-3)
    # C_bot = A_bl * B_top + A_br * B_bot
    # Each product is 1x2x5
    
    # A_bl * B_top: A[2, 0:2] * B[0:2, 0:5] -> C[2, 0:5]
    # Split B_top into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    # 1x2x2 + 1x2x2 + 1x2x1
    add_1x2x2(2, 0, 0, 2, 0)         # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_1x2x2(2, 0, 2, 2, 2)         # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_1x2x1(2, 0, 4, 2, 4)         # A[2,0:2] * B[0:2,4:5] -> C[2,4:5]
    
    # A_br * B_bot: A[2, 2:4] * B[2:4, 0:5] -> C[2, 0:5]
    # Split B_bot into B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    add_1x2x2(2, 2, 0, 2, 0)         # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    add_1x2x2(2, 2, 2, 2, 2)         # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    add_1x2x1(2, 2, 4, 2, 4)         # A[2,2:4] * B[2:4,4:5] -> C[2,4:5]
    
    # Now convert dictionaries to lists
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U_dict.items():
        for (c, coef) in entries:
            U[i * m + j][c] = coef
    for (j, k), entries in V_dict.items():
        for (c, coef) in entries:
            V[j * p + k][c] = coef
    for (k, i), entries in W_dict.items():
        for (c, coef) in entries:
            W[k * n + i][c] = coef
    
    return U, V, W, rank


def _build_schoolbook():
    """Build the schoolbook rank-60 decomposition as fallback."""
    n, m, p = 3, 4, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def construct():
    deadline = time.time() + 140.0
    
    # Try the block-structured Strassen decomposition (rank 42)
    U, V, W, rank = _build_decomposition()
    
    if time.time() < deadline and _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 60)
    U, V, W, rank = _build_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
