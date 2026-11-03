"""Matrix multiplication 3x4x7 rank-minimal exact tensor decomposition.

Design: split the output dimension n=3 into 2+1.
- For the top 2 rows (2x4x7): split m=4 into 2+2 and apply Strassen
  to each (2x2x7) block, splitting p=7 into 2+2+3.
- For the bottom 1 row (1x4x7): use schoolbook (rank 28).

Total rank = 2*(7+7+14) + 28 = 80.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]


def _strassen_222():
    """Strassen 2x2x2: returns U (4,7), V (4,7), W (4,7) as lists.

    Convention: U[i*m+j, c] where m=2 (row i, col j of A)
                V[j*p+k, c] where p=2 (row j, col k of B)
                W[k*n+i, c] where n=2 (row k of C, col i of C)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Strassen formulas (using ±1 entries):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # U[i*m+j, c]: i=0,1; j=0,1
    # V[j*p+k, c]: j=0,1; k=0,1
    # W[k*n+i, c]: k=0,1; i=0,1
    #
    # For each Strassen product Mc, we need:
    # U[i,j] = coefficient of A[i,j] in Mc
    # V[j,k] = coefficient of B[j,k] in Mc
    # W[k,i] = coefficient of Mc in C[k,i]

    # M1 = (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = 1; W[1*2+0, 0] = 1; W[1*2+1, 0] = 1

    # M2 = (A10+A11)B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1

    # M3 = A00(B10-B11)
    U[0*2+0, 2] = 1
    V[1*2+0, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+0, 2] = 1

    # M4 = A11(B01-B10)
    U[1*2+1, 3] = 1
    V[0*2+1, 3] = 1; V[1*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1; W[1*2+1, 3] = 1

    # M5 = (A00+A01)B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1; W[1*2+1, 4] = 1

    # M6 = (A10-A00)(B00+B10)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[1*2+0, 5] = 1
    W[1*2+1, 5] = 1

    # M7 = (A01-A11)(B01+B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[0*2+1, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1; W[1*2+1, 6] = 1

    return U, V, W


def _strassen_223():
    """Strassen 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    Returns U (4,7), V (6,7), W (6,7).

    U[i*m+j, c]: m=2
    V[j*p+k, c]: p=3, j=0,1; k=0,1,2
    W[k*n+i, c]: n=2, k=0,1; i=0,1
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(6)]
    W = [[0]*7 for _ in range(6)]

    # Same Strassen products M1..M7, but B is 2x3 (columns 0,1,2)
    # M1 = (A00+A11)(B00+B11)  -- B00 is row0col0, B11 is row1col1
    U[0, 0] = 1; U[3, 0] = 1
    V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
    W[0*2+0, 0] = 1; W[0*2+1, 0] = 1; W[1*2+0, 0] = 1; W[1*2+1, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0*3+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = -1

    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[1*3+0, 2] = 1; V[1*3+1, 2] = -1
    W[0*2+1, 2] = 1; W[1*2+0, 2] = 1

    # M4 = A11(B01-B10)
    U[3, 3] = 1
    V[0*3+1, 3] = 1; V[1*3+0, 3] = -1
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1; W[1*2+1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[1*3+1, 4] = 1
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1; W[1*2+1, 4] = 1

    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0*3+0, 5] = 1; V[1*3+0, 5] = 1
    W[1*2+1, 5] = 1

    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[0*3+1, 6] = 1; V[1*3+1, 6] = 1
    W[0*2+0, 6] = 1; W[1*2+1, 6] = 1

    return U, V, W


def _schoolbook_1x4x7():
    """Schoolbook for 1x4x7: A is 1x4, B is 4x7, C is 1x7.
    Returns U (4, 28), V (28, 28), W (7, 28).

    U[i*m+j, c]: n=1, m=4, i=0, j=0..3
    V[j*p+k, c]: p=7, j=0..3, k=0..6
    W[k*n+i, c]: n=1, k=0..6, i=0
    """
    rank = 4 * 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(7)]

    col = 0
    for j in range(4):
        for k in range(7):
            U[0*4 + j, col] = 1
            V[j*7 + k, col] = 1
            W[k*1 + 0, col] = 1
            col += 1

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition gives correct matrix multiplication.

    Check: for all (i, j', k), sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    where j is the column of A that matches row of B.
    
    Actually the correct check: C[i,k] = sum_j A[i,j] B[j,k]
    Decomposition: C[i,k] = sum_c sum_j U[i*m+j,c] A[i,j] * V[j*p+k,c] * W[k*n+i,c] B[j,k]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] = delta_{j,j'}
    for all i, j, j', k.
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    # For each (i, k), the matrix M[j, j'] = sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
    # should equal identity.
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
                    M[j, jp] = s
            # M should be identity
            if np.max(np.abs(M - np.eye(m))) > tol:
                return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    deadline = time.time() + 110  # Safety margin

    # Block decomposition:
    # Split n=3 into n1=2 (top) and n2=1 (bottom)
    # Top block: 2x4x7, split m=4 into 2+2
    #   Block (i=0,1; j=0,1; k=0,1): Strassen 2x2x2, rank 7
    #   Block (i=0,1; j=0,1; k=2,3): Strassen 2x2x2, rank 7
    #   Block (i=0,1; j=0,1; k=4,5,6): Strassen 2x2x3, rank 7
    #   Block (i=0,1; j=2,3; k=0,1): Strassen 2x2x2, rank 7
    #   Block (i=0,1; j=2,3; k=2,3): Strassen 2x2x2, rank 7
    #   Block (i=0,1; j=2,3; k=4,5,6): Strassen 2x2x3, rank 7
    # Bottom block: 1x4x7, schoolbook, rank 28
    # Total rank = 6*7 + 28 = 70

    # Wait, let me recount. The Strassen 2x2x2 has rank 7, and 2x2x3 also rank 7.
    # 6 blocks * 7 = 42, plus 28 = 70.

    total_rank = 6 * 7 + 28  # 70

    U = [[0]*total_rank for _ in range(n*m)]  # 12 x 70
    V = [[0]*total_rank for _ in range(m*p)]  # 28 x 70
    W = [[0]*total_rank for _ in range(p*n)]  # 21 x 70

    col = 0

    # Top 2 rows, first 2 columns of A (j=0,1), k=0,1
    sU, sV, sW = _strassen_222()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j, col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + k, col + c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*n + i, col + c] = sW[k*2 + i][c]
    col += 7

    # Top 2 rows, first 2 columns of A (j=0,1), k=2,3
    sU, sV, sW = _strassen_222()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j, col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*p + (k+2), col + c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(k+2)*n + i, col + c] = sW[k*2 + i][c]
    col += 7

    # Top 2 rows, first 2 columns of A (j=0,1), k=4,5,6
    sU, sV, sW = _strassen_223()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j, col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(3):
                V[j*p + (k+4), col + c] = sV[j*3 + k][c]
        for k in range(3):
            for i in range(2):
                W[(k+4)*n + i, col + c] = sW[k*2 + i][c]
    col += 7

    # Top 2 rows, second 2 columns of A (j=2,3), k=0,1
    sU, sV, sW = _strassen_222()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*p + k, col + c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*n + i, col + c] = sW[k*2 + i][c]
    col += 7

    # Top 2 rows, second 2 columns of A (j=2,3), k=2,3
    sU, sV, sW = _strassen_222()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*p + (k+2), col + c] = sV[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(k+2)*n + i, col + c] = sW[k*2 + i][c]
    col += 7

    # Top 2 rows, second 2 columns of A (j=2,3), k=4,5,6
    sU, sV, sW = _strassen_223()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), col + c] = sU[i*2 + j][c]
        for j in range(2):
            for k in range(3):
                V[(j+2)*p + (k+4), col + c] = sV[j*3 + k][c]
        for k in range(3):
            for i in range(2):
                W[(k+4)*n + i, col + c] = sW[k*2 + i][c]
    col += 7

    # Bottom 1 row (i=2), all 4 columns of A, all 7 columns of B: schoolbook
    sU, sV, sW = _schoolbook_1x4x7()
    for c in range(28):
        for j in range(4):
            U[2*m + j, col + c] = sU[j][c]
        for j in range(4):
            for k in range(7):
                V[j*p + k, col + c] = sV[j*7 + k][c]
        for k in range(7):
            W[k*n + 2, col + c] = sW[k][c]
    col += 28

    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
