"""Strassen-based decomposition for 3x3x8 matrix multiplication.

Uses rank-7 Strassen for the 2x2x2 sub-block (i,j in {0,1}, k in {0,1}),
then schoolbook for the remaining 64 terms. Total rank: 7 + 64 = 71.
"""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 71
    U = [[0.0] * rank for _ in range(n * m)]   # 9 x 71
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 71
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 71

    # ---- Strassen 2x2x2 sub-block: i,j in {0,1}, k in {0,1} ----
    # The 8 schoolbook terms for this sub-block are replaced by 7 Strassen terms.
    #
    # Strassen 2x2x2 (for A 2x2, B 2x2, C 2x2):
    #   m1 = (A00+A11)*(B00+B11)
    #   m2 = (A10+A11)*B00
    #   m3 = A00*(B10-B11)
    #   m4 = A11*(B01-B00)
    #   m5 = (A00+A01)*B11
    #   m6 = (A10-A00)*(B00+B10)
    #   m7 = (A01-A11)*(B01+B11)
    #
    #   C00 = m1 + m4 - m5 + m7
    #   C01 = m3 + m5
    #   C10 = m2 + m4
    #   C11 = m1 - m2 + m3 + m6
    #
    # In tensor decomposition terms, each Strassen term is a rank-1 term
    # with coefficients that encode the linear combinations.
    #
    # For the sub-block (i,j,k) in {0,1}^3, the tensor entry T[i,j,k] = 1
    # means A[i,j]*B[j,k] contributes to C[i,k].
    #
    # Strassen term r has:
    #   U[i*m+j, r] = coefficient of A[i,j] in the first factor
    #   V[j*p+k, r] = coefficient of B[j,k] in the second factor
    #   W[k*n+i, r] = coefficient for output C[i,k]
    #
    # Let me define each Strassen term explicitly.
    #
    # Term 1: m1 = (A00+A11)*(B00+B11)
    #   First factor: A00 + A11  => U[0,0]=1, U[4,0]=1 (A[0,0], A[1,1])
    #   Second factor: B00 + B11 => V[0,0]=1, V[9,0]=1 (B[0,0], B[1,1])
    #   Output: contributes to C00 and C11
    #     C00 gets +m1, C11 gets +m1
    #     W[0,0]=1 (C[0,0], k=0,i=0), W[3,0]=1 (C[1,1], k=1,i=1)
    #     Wait, W index is k*n+i. For C[0,0]: k=0,i=0 => W[0]. For C[1,1]: k=1,i=1 => W[1*3+1]=W[4].
    #     So W[0,0]=1, W[4,0]=1.

    # Term 1: m1 = (A00+A11)*(B00+B11)
    U[0][0] = 1.0; U[4][0] = 1.0
    V[0][0] = 1.0; V[9][0] = 1.0
    W[0][0] = 1.0; W[4][0] = 1.0

    # Term 2: m2 = (A10+A11)*B00
    #   First: A10 + A11 => U[3,1]=1, U[4,1]=1
    #   Second: B00 => V[0,2]=1
    #   Output: C10 gets +m2. C[1,0]: k=0,i=1 => W[0*3+1]=W[1]
    U[3][1] = 1.0; U[4][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0

    # Term 3: m3 = A00*(B10-B11)
    #   First: A00 => U[0,2]=1
    #   Second: B10 - B11 => V[8,2]=1, V[9,2]=-1
    #   Output: C01 gets +m3. C[0,1]: k=1,i=0 => W[1*3+0]=W[3]
    U[0][2] = 1.0
    V[8][2] = 1.0; V[9][2] = -1.0
    W[3][2] = 1.0

    # Term 4: m4 = A11*(B01-B00)
    #   First: A11 => U[4,3]=1
    #   Second: B01 - B00 => V[1,3]=1, V[0,3]=-1
    #   Output: C00 gets +m4, C10 gets +m4
    #   C[0,0]: W[0], C[1,0]: W[1]
    U[4][3] = 1.0
    V[1][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0

    # Term 5: m5 = (A00+A01)*B11
    #   First: A00 + A01 => U[0,4]=1, U[1,4]=1
    #   Second: B11 => V[9,4]=1
    #   Output: C00 gets -m5, C01 gets +m5
    #   C[0,0]: W[0], C[0,1]: W[3]
    U[0][4] = 1.0; U[1][4] = 1.0
    V[9][4] = 1.0
    W[0][4] = -1.0; W[3][4] = 1.0

    # Term 6: m6 = (A10-A00)*(B00+B10)
    #   First: A10 - A00 => U[3,5]=1, U[0,5]=-1
    #   Second: B00 + B10 => V[0,5]=1, V[8,5]=1
    #   Output: C11 gets +m6. C[1,1]: k=1,i=1 => W[4]
    U[3][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[8][5] = 1.0
    W[4][5] = 1.0

    # Term 7: m7 = (A01-A11)*(B01+B11)
    #   First: A01 - A11 => U[1,6]=1, U[4,6]=-1
    #   Second: B01 + B11 => V[1,6]=1, V[9,6]=1
    #   Output: C00 gets +m7. C[0,0]: W[0]
    U[1][6] = 1.0; U[4][6] = -1.0
    V[1][6] = 1.0; V[9][6] = 1.0
    W[0][6] = 1.0

    # ---- Schoolbook terms for all other (i,j,k) ----
    # All (i,j,k) where NOT (i<2 and j<2 and k<2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
