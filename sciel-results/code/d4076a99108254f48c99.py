"""Matrix multiplication 2x4x7 — Strassen-based block decomposition."""
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def _strassen_2x2():
    """Return (U, V, W) as lists of 7 rank-1 terms for 2x2x2 multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    # Strassen's 7 products for (2x2)*(2x2)->(2x2):
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y10-y11)
    # M4 = x11*(y01-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    #
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 + M5
    # C10 = M2 + M3 - M1 + M6
    # C11 = -M3 + M4 + M6 + M7
    #
    # U[i*2+j] for each M:
    # M1: x00,x11 -> U[0]=1, U[3]=1
    # M2: x10,x11 -> U[2]=1, U[3]=1
    # M3: x00      -> U[0]=1
    # M4: x11      -> U[3]=1
    # M5: x00,x01  -> U[0]=1, U[1]=1
    # M6: x10,-x00 -> U[2]=1, U[0]=-1
    # M7: x01,-x11 -> U[1]=1, U[3]=-1
    #
    # V[j*2+k] for each M:
    # M1: y00,y11 -> V[0]=1, V[3]=1
    # M2: y00      -> V[0]=1
    # M3: y10,-y11 -> V[2]=1, V[3]=-1
    # M4: y01,-y00 -> V[1]=1, V[0]=-1
    # M5: y11      -> V[3]=1
    # M6: y00,y10  -> V[0]=1, V[2]=1
    # M7: y01,y11  -> V[1]=1, V[3]=1
    #
    # W[k*2+i] for each M (output C[i,k]):
    # M1: C00(-1? no +1), C10(-1) -> W[0*2+0]=1, W[0*2+1]=-1
    #    Wait: C00 = M1+... so M1 contributes +1 to C00 (k=0,i=0) -> W[0]=1
    #         C10 = ...-M1+... so M1 contributes -1 to C10 (k=0,i=1) -> W[1]=-1
    # M2: C01(+1), C10(+1) -> W[1*2+0]=1 (k=1,i=0), W[0*2+1]=1 (k=0,i=1)
    # M3: C01(+1), C10(+1), C11(-1) -> W[2]=1(k=1,i=0), W[1]=1(k=0,i=1), W[3]=-1(k=1,i=1)
    # M4: C00(+1), C11(+1) -> W[0]=1(k=0,i=0), W[3]=1(k=1,i=1)
    # M5: C00(-1), C01(+1) -> W[0]=-1(k=0,i=0), W[2]=1(k=1,i=0)
    # M6: C10(+1), C11(+1) -> W[1]=1(k=0,i=1), W[3]=1(k=1,i=1)
    # M7: C00(+1), C11(+1) -> W[0]=1(k=0,i=0), W[3]=1(k=1,i=1)

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # Term 0 (M1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1

    # Term 1 (M2)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[1][1] = 1

    # Term 2 (M3)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[1][2] = 1; W[3][2] = -1

    # Term 3 (M4)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[3][3] = 1

    # Term 4 (M5)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # Term 5 (M6)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[1][5] = 1; W[3][5] = 1

    # Term 6 (M7)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1

    return U, V, W


def _strassen_2x2x1():
    """Return (U, V, W) for 2x2 * 2x1 -> 2x1 multiplication, rank 3 (Strassen-like).
    Actually 2x2 times 2x1: C_i = sum_j x_ij y_j. This is just bilinear, rank = 4 schoolbook.
    But we can use: M1 = x00*y0, M2 = x01*y1, M3 = (x00+x01)*(y0+y1).
    Then C0 = M1 + M3 - (x01*y0) ... hmm that's not right.
    Actually for 2x2 * 2x1, the tensor has rank exactly 4 (no known improvement).
    Let me just use schoolbook rank 4.
    """
    # Schoolbook: 4 terms
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    # U[i*2+j], V[j*1+k]=V[j], W[k*2+i]=W[i] (since p=1, k=0 always)
    # Actually p=1: V shape (m*p, rank)=(2,4), W shape (p*n, rank)=(2,4)
    # V[j*1+0] = V[j], W[0*2+i] = W[i]
    for i in range(2):
        for j in range(2):
            c = i*2+j
            U[c][c] = 1
            V[j][c] = 1
            W[i][c] = 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7

    # Block decomposition:
    # A is 2x4: split columns into 2x2 and 2x2 blocks -> A = [A0 | A1]
    # B is 4x7: split rows into 2+2+1: B = [B0; B1; B2] where B0 is 2x7, B1 is 2x7, B2 is 1x7
    # Actually m=4, so we split the inner dimension into blocks.
    #
    # Let's think of it as: C = A * B where A is 2x4, B is 4x7.
    # Split A into two 2x2 column blocks: A = [A_L | A_R]
    # Split B into three row blocks: B = [B_L; B_M; B_R] where sizes are 2, 2, 1
    # Then C = A_L * B_L + A_R * B_M + A_R_extra... 
    #
    # Wait, A is 2x4, B is 4x7. The inner dimension is 4.
    # Split into blocks of size 2: two blocks of size 2.
    # A = [A0 | A1] where A0, A1 are 2x2.
    # B = [B0; B1] where B0, B1 are 2x7.
    # C = A0*B0 + A1*B1. Each is 2x2 * 2x7.
    #
    # 2x2 * 2x7: split the 7 columns of B into 2+2+2+1:
    # B = [Bcol0 | Bcol1 | Bcol2 | Bcol3] where first three are 2x2, last is 2x1.
    # A0 * B0 = A0 * [Bcol0 | Bcol1 | Bcol2 | Bcol3]
    #   = A0*Bcol0 + A0*Bcol1 + A0*Bcol2 + A0*Bcol3
    # Each A0*Bcol_j for j=0,1,2 is 2x2 * 2x2 -> Strassen rank 7.
    # A0*Bcol3 is 2x2 * 2x1 -> rank 4 (schoolbook).
    #
    # Total rank = 2 * (3*7 + 4) = 2 * 25 = 50.
    #
    # Similarly for A1.
    #
    # So total rank = 50.
    #
    # Let's construct this carefully.

    # Indices:
    # A[i, j] where i in {0,1} (rows of A, n=2), j in {0,1,2,3} (cols of A, m=4)
    # B[j, k] where j in {0,1,2,3} (rows of B, m=4), k in {0,...,6} (cols of B, p=7)
    # C[i, k] where i in {0,1}, k in {0,...,6}
    #
    # W convention: W[k*n + i, r]
    #
    # U convention: U[i*m + j, r] = U[i*4+j, r]
    # V convention: V[j*p + k, r] = V[j*7+k, r]
    #
    # Block structure:
    # A_L = A[:, 0:2] (columns 0,1), A_R = A[:, 2:4] (columns 2,3)
    # B_L = B[0:2, :] (rows 0,1), B_R = B[2:4, :] (rows 2,3)
    #
    # C = A_L * B_L + A_R * B_R
    #
    # For A_L * B_L: A_L is 2x2, B_L is 2x7.
    #   Split B_L columns into 4 blocks: cols 0-1, 2-3, 4-5, 6
    #   B_L[:, 0:2] (2x2), B_L[:, 2:4] (2x2), B_L[:, 4:6] (2x2), B_L[:, 6:7] (2x1)
    #
    # For each 2x2 * 2x2 block: 7 Strassen terms.
    # For each 2x2 * 2x1 block: 4 schoolbook terms.
    #
    # Total per side: 3*7 + 4 = 25 terms.
    # Two sides: 50 terms.

    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 8 x 50
    V = [[0] * rank for _ in range(m * p)]   # 28 x 50
    W = [[0] * rank for _ in range(p * n)]   # 14 x 50

    sU, sV, sW = _strassen_2x2()  # each 4x7
    # Schoolbook 2x2*2x1
    scU = [[0]*4 for _ in range(4)]
    scV = [[0]*4 for _ in range(2)]
    scW = [[0]*4 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            c = i * 2 + j
            scU[c][c] = 1
            scV[j][c] = 1
            scW[i][c] = 1

    col = 0

    # For side s in {0, 1}: s=0 -> A_L (cols 0,1), B_L (rows 0,1); s=1 -> A_R (cols 2,3), B_R (rows 2,3)
    for s in range(2):
        a_col_offset = s * 2  # 0 or 2
        b_row_offset = s * 2  # 0 or 2

        # Three 2x2 * 2x2 blocks (column pairs 0-1, 2-3, 4-5 of B)
        for bcol_pair in range(3):
            k_offset = bcol_pair * 2  # 0, 2, 4
            for t in range(7):
                # U: for local (i, j) where j in {0,1} maps to global col a_col_offset + j
                for i in range(2):
                    for j in range(2):
                        u_val = sU[i * 2 + j][t]
                        if u_val != 0:
                            U[i * m + a_col_offset + j][col] += u_val
                # V: for local (j, k) where j in {0,1} maps to global row b_row_offset + j, k in {0,1} maps to global col k_offset + k
                for j in range(2):
                    for k in range(2):
                        v_val = sV[j * 2 + k][t]
                        if v_val != 0:
                            V[(b_row_offset + j) * p + (k_offset + k)][col] += v_val
                # W: for local (k, i) where k in {0,1} maps to global col k_offset + k, i in {0,1}
                for k in range(2):
                    for i in range(2):
                        w_val = sW[k * 2 + i][t]
                        if w_val != 0:
                            W[(k_offset + k) * n + i][col] += w_val
                col += 1

        # One 2x2 * 2x1 block (column 6 of B)
        k_offset = 6
        for t in range(4):
            for i in range(2):
                for j in range(2):
                    u_val = scU[i * 2 + j][t]
                    if u_val != 0:
                        U[i * m + a_col_offset + j][col] += u_val
            for j in range(2):
                v_val = scV[j][t]
                if v_val != 0:
                    V[(b_row_offset + j) * p + k_offset][col] += v_val
            for i in range(2):
                w_val = scW[i][t]
                if w_val != 0:
                    W[k_offset * n + i][col] += w_val
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
