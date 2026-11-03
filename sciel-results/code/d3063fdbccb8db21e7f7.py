import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank


def _combine(U_list, V_list, W_list, i_start, j_start, k_start, n, m, p):
    """
    Combine local decompositions into global U, V, W.
    Each local decomposition has dimensions (n_i, m_i, p_i).
    """
    n_i, m_i, p_i = n, m, p
    # Actually we need to know the local dimensions for indexing.
    # Let's pass local dimensions explicitly.
    pass


def _block_decompose(n, m, p, n1, m1, p1):
    """
    Decompose matrix multiplication of size n x m * m x p -> n x p
    using a 2x2 block structure where:
    A = [A11 A12]  (n1 x m1, n1 x (m-m1))
    B = [B11 B12]  (m1 x p1, (m-m1) x p1)  -- Wait, B is m x p.
    So B is split into rows: top m1, bottom m-m1.
    B11: m1 x p1, B12: m1 x (p-p1)
    B21: (m-m1) x p1, B22: (m-m1) x (p-p1)
    C = [C11 C12]
    C11 = A11 B11 + A12 B21
    C12 = A11 B12 + A12 B22
    """
    n2 = n - n1
    m2 = m - m1
    p2 = p - p1

    # Sub-problems:
    # 1. A11 (n1 x m1) * B11 (m1 x p1) -> C11_part1 (n1 x p1)
    # 2. A12 (n1 x m2) * B21 (m2 x p1) -> C11_part2 (n1 x p1)
    # 3. A11 (n1 x m1) * B12 (m1 x p2) -> C12_part1 (n1 x p2)
    # 4. A12 (n1 x m2) * B22 (m2 x p2) -> C12_part2 (n1 x p2)

    # We can use Strassen for 2x2x2 if dimensions allow, otherwise schoolbook.
    # Let's define a helper that returns U, V, W for a specific sub-matrix multiplication.

    def _sub_decompose(nn, mm, pp):
        # Use Strassen if nn=mm=pp=2, else schoolbook
        if nn == 2 and mm == 2 and pp == 2:
            return _strassen_2x2()
        else:
            return _schoolbook(nn, mm, pp)

    U1, V1, W1, r1 = _sub_decompose(n1, m1, p1)  # A11 * B11
    U2, V2, W2, r2 = _sub_decompose(n1, m2, p1)  # A12 * B21
    U3, V3, W3, r3 = _sub_decompose(n1, m1, p2)  # A11 * B12
    U4, V4, W4, r4 = _sub_decompose(n1, m2, p2)  # A12 * B22

    total_rank = r1 + r2 + r3 + r4

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0

    # Sub-problem 1: A11 (rows 0..n1, cols 0..m1) * B11 (rows 0..m1, cols 0..p1) -> C11 (rows 0..n1, cols 0..p1)
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[i * m + j][col + c] = U1[i * m1 + j][c]
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[j * p + k][col + c] = V1[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[k * n + i][col + c] = W1[k * n1 + i][c]
    col += r1

    # Sub-problem 2: A12 (rows 0..n1, cols m1..m) * B21 (rows m1..m, cols 0..p1) -> C11 (rows 0..n1, cols 0..p1)
    for i in range(n1):
        for j in range(m2):
            for c in range(r2):
                U[i * m + (m1 + j)][col + c] = U2[i * m2 + j][c]
    for j in range(m2):
        for k in range(p1):
            for c in range(r2):
                V[(m1 + j) * p + k][col + c] = V2[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r2):
                W[k * n + i][col + c] += W2[k * n1 + i][c]  # Add to C11
    col += r2

    # Sub-problem 3: A11 (rows 0..n1, cols 0..m1) * B12 (rows 0..m1, cols p1..p) -> C12 (rows 0..n1, cols p1..p)
    for i in range(n1):
        for j in range(m1):
            for c in range(r3):
                U[i * m + j][col + c] = U3[i * m1 + j][c]
    for j in range(m1):
        for k in range(p2):
            for c in range(r3):
                V[j * p + (p1 + k)][col + c] = V3[j * p2 + k][c]
    for k in range(p2):
        for i in range(n1):
            for c in range(r3):
                W[(p1 + k) * n + i][col + c] = W3[k * n1 + i][c]
    col += r3

    # Sub-problem 4: A12 (rows 0..n1, cols m1..m) * B22 (rows m1..m, cols p1..p) -> C12 (rows 0..n1, cols p1..p)
    for i in range(n1):
        for j in range(m2):
            for c in range(r4):
                U[i * m + (m1 + j)][col + c] = U4[i * m2 + j][c]
    for j in range(m2):
        for k in range(p2):
            for c in range(r4):
                V[(m1 + j) * p + (p1 + k)][col + c] = V4[j * p2 + k][c]
    for k in range(p2):
        for i in range(n1):
            for c in range(r4):
                W[(p1 + k) * n + i][col + c] += W4[k * n1 + i][c]  # Add to C12
    col += r4

    return U, V, W, total_rank


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    A = [a b; c d], B = [e f; g h]
    C = [p+q+t-r, q+s; q+s, p+q-r] where:
    p = (a+d)(e+h)
    q = (c+d)(g+h)
    r = a(f-h)
    s = d(g-e)
    t = (a+b)(h+e) -- wait, standard Strassen:
    m1 = (a+d)(e+h)
    m2 = (c+d)(e+f)
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)(h)
    m6 = (c-d)(e)
    m7 = (b-d)(f+g)
    C11 = m1 + m4 - m5 + m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m2 + m3 + m6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3

    # m1 = (a+d)(e+h)
    # a: U[0], d: U[3]
    # e: V[0], h: V[3]
    # C11: W[0]
    U[0][0] = 1.0
    U[3][0] = 1.0
    V[0][0] = 1.0
    V[3][0] = 1.0
    W[0][0] = 1.0

    # m2 = (c+d)(e+f)
    # c: U[2], d: U[3]
    # e: V[0], f: V[1]
    # C21: W[2], C22: W[3] (with -1)
    U[2][1] = 1.0
    U[3][1] = 1.0
    V[0][1] = 1.0
    V[1][1] = 1.0
    W[2][1] = 1.0
    W[3][1] = -1.0

    # m3 = a(f-h)
    # a: U[0]
    # f: V[1], h: V[3]
    # C12: W[1], C22: W[3]
    U[0][2] = 1.0
    V[1][2] = 1.0
    V[3][2] = -1.0
    W[1][2] = 1.0
    W[3][2] = 1.0

    # m4 = d(g-e)
    # d: U[3]
    # g: V[2], e: V[0]
    # C11: W[0], C21: W[2]
    U[3][3] = 1.0
    V[2][3] = 1.0
    V[0][3] = -1.0
    W[0][3] = 1.0
    W[2][3] = 1.0

    # m5 = (a+b)h
    # a: U[0], b: U[1]
    # h: V[3]
    # C11: W[0] (-1), C12: W[1]
    U[0][4] = 1.0
    U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0
    W[1][4] = 1.0

    # m6 = (c-d)e
    # c: U[2], d: U[3]
    # e: V[0]
    # C22: W[3]
    U[2][5] = 1.0
    U[3][5] = -1.0
    V[0][5] = 1.0
    W[3][5] = 1.0

    # m7 = (b-d)(f+g)
    # b: U[1], d: U[3]
    # f: V[1], g: V[2]
    # C11: W[0]
    U[1][6] = 1.0
    U[3][6] = -1.0
    V[1][6] = 1.0
    V[2][6] = 1.0
    W[0][6] = 1.0

    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k,r] = U[i*m+j, :] dot V[j*p+k, :] * W[k*n+r, :]
    # We want T[i,j,k,r] = delta_{i,r} delta_{j,k}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    # 4x5x5
    # Split n=4 into n1=2, n2=2
    # Split m=5 into m1=2, m2=3
    # Split p=5 into p1=2, p2=3
    U, V, W, rank = _block_decompose(n, m, p, 2, 2, 2)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
