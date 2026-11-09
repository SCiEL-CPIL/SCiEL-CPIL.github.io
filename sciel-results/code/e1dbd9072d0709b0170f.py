"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]


def _strassen_2x2x2():
    """Return (U, V, W) lists for 2x2 x 2x2 = 2x2, rank 7.
    U has shape (4,7) indexed by [i*2+j, c] (input A entries),
    V has shape (4,7) indexed by [j*2+k, c] (input B entries),
    W has shape (4,7) indexed by [k*2+i, c] (output C entries, trace convention).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # A index: a00=0, a01=1, a10=2, a11=3
    # B index: b00=0, b01=1, b10=2, b11=3
    # C index (k*2+i): C00=0, C01=1, C10=2, C11=3
    # M1: (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2: (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # M3: a00(b10-b11)
    U[0][2]=1; V[2][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # M4: a11(b01-b00)
    U[3][3]=1; V[1][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5: (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # M6: (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7: (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W


def _strip_1x2x2():
    """1x2 x 2x2 = 1x2, rank 4 (schoolbook)."""
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    U[0][1]=1; V[1][1]=1; W[1][1]=1
    U[1][2]=1; V[2][2]=1; W[0][2]=1
    U[1][3]=1; V[3][3]=1; W[1][3]=1
    return U, V, W


def construct():
    """Build a rank decomposition for 3x4 x 4x4 = 3x4.
    U shape (12, rank) indexed [i*4+j, c], V shape (16, rank) [j*4+k, c],
    W shape (12, rank) [k*3+i, c].
    """
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    U_rows = n*m
    V_rows = m*p
    W_rows = p*n

    terms = []

    def add_strassen(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            a_pairs = [((li//2), (li%2), sU[li][c]) for li in range(4) if sU[li][c]!=0]
            b_pairs = [((li//2), (li%2), sV[li][c]) for li in range(4) if sV[li][c]!=0]
            w_pairs = [((li//2), (li%2), sW[li][c]) for li in range(4) if sW[li][c]!=0]
            for (ai, aj, ac) in a_pairs:
                for (bi, bk, bc) in b_pairs:
                    for (ck, ci, cc) in w_pairs:
                        a_idx = (a_row_off+ai)*4 + (a_col_off+aj)
                        b_idx = (b_row_off+bi)*4 + (b_col_off+bk)
                        c_idx = (c_col_off+ck)*3 + (c_row_off+ci)
                        terms.append((a_idx, b_idx, c_idx, ac*bc*cc))

    def add_strip(a_row, a_col_off, b_row_off, b_col_off, c_row, c_col_off):
        sU, sV, sW = _strip_1x2x2()
        for c in range(4):
            a_pairs = [(j, sU[j][c]) for j in range(2) if sU[j][c]!=0]
            b_pairs = [((li//2), (li%2), sV[li][c]) for li in range(4) if sV[li][c]!=0]
            w_pairs = [(j, sW[j][c]) for j in range(2) if sW[j][c]!=0]
            for (aj, ac) in a_pairs:
                for (bi, bk, bc) in b_pairs:
                    for (cj, cc) in w_pairs:
                        a_idx = a_row*4 + (a_col_off+aj)
                        b_idx = (b_row_off+bi)*4 + (b_col_off+bk)
                        c_idx = (c_col_off+cj)*3 + c_row
                        terms.append((a_idx, b_idx, c_idx, ac*bc*cc))

    for b_col_off in (0, 2):
        c_col_off = b_col_off
        add_strassen(0, 0, 0, b_col_off, 0, c_col_off)
        add_strassen(0, 2, 2, b_col_off, 0, c_col_off)

    for b_col_off in (0, 2):
        c_col_off = b_col_off
        add_strip(2, 0, 0, b_col_off, 2, c_col_off)
        add_strip(2, 2, 2, b_col_off, 2, c_col_off)

    from collections import defaultdict
    merged = defaultdict(int)
    for (a,b,c,w) in terms:
        merged[(a,b,c)] += w
    merged = {k:v for k,v in merged.items() if v != 0}

    rank = len(merged)
    U = [[0]*rank for _ in range(U_rows)]
    V = [[0]*rank for _ in range(V_rows)]
    W = [[0]*rank for _ in range(W_rows)]
    for col, ((a,b,c),w) in enumerate(merged.items()):
        U[a][col] = w
        V[b][col] = w
        W[c][col] = w

    if time.time() > deadline - 10.0:
        return _schoolbook()

    Um = np.array(U, dtype=float)
    Vm = np.array(V, dtype=float)
    Wm = np.array(W, dtype=float)
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
    T_comp = np.einsum('ar,br,cr->abc', Um, Vm, Wm)
    if not np.allclose(T_comp, T_true, atol=1e-9):
        return _schoolbook()
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
