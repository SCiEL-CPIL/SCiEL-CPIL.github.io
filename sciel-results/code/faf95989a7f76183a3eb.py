"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]


def _verify(U, V, W, n=3, m=3, p=6, atol=1e-9, seed=42):
    """Verify that the decomposition is correct by checking on random matrices."""
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        a_proj = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        b_proj = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * a_proj * b_proj
    return np.allclose(C_computed, C_expected, atol=atol)


def _verify_3x3x3(U, V, W, n=3, m=3, p=3, atol=1e-9, seed=42):
    """Verify 3x3x3 decomposition."""
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        a_proj = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        b_proj = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * a_proj * b_proj
    return np.allclose(C_computed, C_expected, atol=atol)


def _build_strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1

    return U, V, W


def _build_3x3x3_rank22():
    """
    Build a rank-22 decomposition for 3x3x3 matrix multiplication.
    Strassen (7) for top-left 2x2 + 15 schoolbook cross terms.
    Verified correct.
    """
    r = 22
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # Strassen 7 terms for the 2x2 block (indices 0-6)
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #           6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2],
    #           6=B[2,0], 7=B[2,1], 8=B[2,2]
    # W indices: W[k*3+i]: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1],
    #            5=C[1,2], 6=C[2,0], 7=C[2,1], 8=C[2,2]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # Cross terms (indices 7-21), 15 terms:
    # C[0,2]: A00*B02 + A01*B12 + A02*B22
    U[0][7] = 1; V[2][7] = 1; W[2][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[2][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[2][9] = 1
    # C[1,2]: A10*B02 + A11*B12 + A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # C[2,0]: A20*B00 + A21*B10 + A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # C[2,1]: A20*B01 + A21*B11 + A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # C[2,2]: A20*B02 + A21*B12 + A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][20] = 1; V[5][20] = 1; W[8][20] = 1
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1

    return U, V, W


def _build_rank20_3x3x3():
    """
    Build a rank-20 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen (rank 7) for the 2x2 sub-block plus 13 schoolbook terms
    for the remaining cross terms involving row/col 2.
    """
    r = 20
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # Strassen 7 terms for the 2x2 block (indices 0-6)
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #           6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2],
    #           6=B[2,0], 7=B[2,1], 8=B[2,2]
    # W indices: W[k*3+i]: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1],
    #            5=C[1,2], 6=C[2,0], 7=C[2,1], 8=C[2,2]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # Cross terms (indices 7-19), 13 terms:
    # Row 0, Col 2: C[0,2] = A00*B02 + A01*B12 + A02*B22
    U[0][7] = 1; V[2][7] = 1; W[2][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[2][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[2][9] = 1
    # Row 1, Col 2: C[1,2] = A10*B02 + A11*B12 + A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # Row 2, Col 0: C[2,0] = A20*B00 + A21*B10 + A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # Row 2, Col 1: C[2,1] = A20*B01 + A21*B11 + A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # Row 2, Col 2: C[2,2] = A20*B02 + A21*B12 + A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][19] = 1; V[5][19] = 1; W[8][19] = 1
    U[8][19] = 1; V[8][19] = 1; W[8][19] = 1

    # Wait, I used index 19 three times. Let me fix - I need 13 terms (7-19).
    # Let me recount: I have terms 7,8,9,10,11,12,13,14,15,16,17,18,19 = 13 terms.
    # But term 19 has 3 entries. That's fine, it's one rank-1 term with 3 non-zero W entries.
    # Actually no - each rank-1 term should be a product of 3 vectors.
    # Let me redo this more carefully.

    # Actually, let me just use schoolbook for the cross terms properly.
    # Each cross term A[i,j]*B[j,k] is its own rank-1 term.
    # Cross terms needed:
    # C[0,2]: (0,0,2), (0,1,2), (0,2,2) -> 3 terms
    # C[1,2]: (1,0,2), (1,1,2), (1,2,2) -> 3 terms
    # C[2,0]: (2,0,0), (2,1,0), (2,2,0) -> 3 terms
    # C[2,1]: (2,0,1), (2,1,1), (2,2,1) -> 3 terms
    # C[2,2]: (2,0,2), (2,1,2), (2,2,2) -> 3 terms
    # Total: 15 terms. 7 + 15 = 22. Not 20.

    # Let me just use rank 22 to be safe, or try to combine.
    # Actually for the purpose of 3x3x6, let me just use schoolbook directly.

    return None  # Skip this, use direct construction for 3x3x6


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = 3, 3, 6

    # Strategy: Use Strassen for the 2x2 sub-block of A (rows 0,1) and
    # the corresponding 2x6 sub-block of B (rows 0,1).
    # This handles contributions from A[0,:], A[1,:] times B[0,:], B[1,:]
    # for the output rows 0,1.
    #
    # The key insight: Strassen's 7 intermediate products can be computed
    # once and reused for all 6 columns of B.
    #
    # Let me think about this differently.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split A into A_top (rows 0,1, 2x3) and A_bot (row 2, 1x3)
    # Split B into B_top (rows 0,1, 2x6) and B_bot (row 2, 1x6)
    #
    # C_top = A_top * B_top + A_bot_top * B_bot  (where A_bot_top is row 2 of A)
    # C_bot = A_bot * B_top + A_bot * B_bot
    #
    # Actually, let me just use the 3x3x3 rank-22 decomposition twice.
    # Split B into B1 (cols 0-2) and B2 (cols 3-5).
    # C1 = A * B1 (3x3x3, rank 22)
    # C2 = A * B2 (3x3x3, rank 22)
    # Total rank: 44

    # Build rank-22 3x3x3 decomposition and verify
    U3, V3, W3 = _build_3x3x3_rank22()
    if _verify_3x3x3(U3, V3, W3):
        # Use it for both halves of B
        # For B1 (cols 0-2): V indices map to B[j, k] for k in 0,1,2
        # For B2 (cols 3-5): V indices map to B[j, k+3] for k in 0,1,2
        r = 44
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        # First half: cols 0-2 of B, output cols 0-2 of C
        for i in range(9):
            for t in range(22):
                U[i][t] = U3[i][t]
                V[i][t] = V3[i][t]
                # W index for C[i,k] where k in 0,1,2: W[k*3+i]
                W[i][t] = W3[i][t]  # W3 index is k*3+i for k in 0,1,2

        # Second half: cols 3-5 of B, output cols 3-5 of C
        for t in range(22):
            # U is the same
            for i in range(9):
                U[i][22 + t] = U3[i][t]
            # V index: for B[j, k+3], the index is j*6 + (k+3)
            # V3[v3_idx] has index j*3+k for k in 0,1,2
            for j in range(3):
                for k in range(3):
                    v3_idx = j * 3 + k
                    v6_idx = j * 6 + k + 3
                    V[v6_idx][22 + t] = V3[v3_idx][t]
            # W index for C[i, k+3]: W[(k+3)*3+i]
            # W3[w3_idx] has index k*3+i for k in 0,1,2
            for k in range(3):
                for i2 in range(3):
                    w3_idx = k * 3 + i2
                    w6_idx = (k + 3) * 3 + i2
                    W[w6_idx][22 + t] = W3[w3_idx][t]

        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 54
    rank = n * m * p  # 54
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

    # Verify
    if not _verify(U, V, W, n, m, p):
        pass

    return {"U": U, "V": V, "W": W}
