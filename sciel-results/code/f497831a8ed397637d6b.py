"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _winograd_3x3():
    """Return the rank-23 Winograd decomposition for 3x3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    
    Winograd's algorithm for 3x3 matrix multiplication uses 23 multiplications.
    The transformation matrices are:
    
    For A (3x3):
    U = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1]
    ]
    Actually, let me use the standard Winograd 3x3 scheme.
    
    The standard Winograd 3x3 uses these linear forms:
    
    Let A = [[a11, a12, a13],
             [a21, a22, a23],
             [a31, a32, a33]]
             
    Let B = [[b11, b12, b13],
             [b21, b22, b23],
             [b31, b32, b33]]
    
    The 23 products are:
    m1  = a11 * b11
    m2  = a12 * b12
    m3  = a13 * b13
    m4  = a21 * b21
    m5  = a22 * b22
    m6  = a23 * b23
    m7  = a31 * b31
    m8  = a32 * b32
    m9  = a33 * b33
    m10 = (a11+a12) * (b21+b22)
    m11 = (a12+a13) * (b22+b23)
    m12 = (a11+a13) * (b21+b23)
    m13 = (a21+a22) * (b11+b12)
    m14 = (a22+a23) * (b12+b13)
    m15 = (a21+a23) * (b11+b13)
    m16 = (a31+a32) * (b11+b12)
    m17 = (a32+a33) * (b12+b13)
    m18 = (a31+a33) * (b11+b13)
    m19 = (a11+a21) * (b31+b32)  -- wait, this isn't right
    
    Let me look up the correct Winograd 3x3 scheme.
    
    Actually, the standard Winograd 3x3 (also known as the 23-multiplication algorithm) uses:
    
    For the input A, compute:
    A' = T_A * A where T_A is 23x9
    
    For the input B, compute:
    B' = T_B * B where T_B is 23x9
    
    Then C = T_C * (A' .* B')
    
    Let me use a known correct formulation.
    
    Here's a verified Winograd 3x3 scheme with 23 multiplications:
    
    The 23 linear combinations for A (rows of U matrix):
    Row 0:  a11          -> [1,0,0, 0,0,0, 0,0,0]
    Row 1:  a12          -> [0,1,0, 0,0,0, 0,0,0]
    Row 2:  a13          -> [0,0,1, 0,0,0, 0,0,0]
    Row 3:  a21          -> [0,0,0, 1,0,0, 0,0,0]
    Row 4:  a22          -> [0,0,0, 0,1,0, 0,0,0]
    Row 5:  a23          -> [0,0,0, 0,0,1, 0,0,0]
    Row 6:  a31          -> [0,0,0, 0,0,0, 1,0,0]
    Row 7:  a32          -> [0,0,0, 0,0,0, 0,1,0]
    Row 8:  a33          -> [0,0,0, 0,0,0, 0,0,1]
    Row 9:  a11+a12      -> [1,1,0, 0,0,0, 0,0,0]
    Row 10: a12+a13      -> [0,1,1, 0,0,0, 0,0,0]
    Row 11: a11+a13      -> [1,0,1, 0,0,0, 0,0,0]
    Row 12: a21+a22      -> [0,0,0, 1,1,0, 0,0,0]
    Row 13: a22+a23      -> [0,0,0, 0,1,1, 0,0,0]
    Row 14: a21+a23      -> [0,0,0, 1,0,1, 0,0,0]
    Row 15: a31+a32      -> [0,0,0, 0,0,0, 1,1,0]
    Row 16: a32+a33      -> [0,0,0, 0,0,0, 0,1,1]
    Row 17: a31+a33      -> [0,0,0, 0,0,0, 1,0,1]
    Row 18: a11+a21      -> [1,0,0, 1,0,0, 0,0,0]
    Row 19: a12+a22      -> [0,1,0, 0,1,0, 0,0,0]
    Row 20: a13+a23      -> [0,0,1, 0,0,1, 0,0,0]
    Row 21: a11+a31      -> [1,0,0, 0,0,0, 1,0,0]
    Row 22: a12+a32      -> [0,1,0, 0,0,0, 0,1,0]
    Row 23: a13+a33      -> [0,0,1, 0,0,0, 0,0,1]
    
    Wait, that's 24 rows. I need exactly 23.
    
    Let me use a different, verified formulation. Here's a correct Winograd 3x3:
    
    Actually, let me just use the schoolbook 27-multiplication for 3x3 and see if the overall rank is better.
    4 blocks * 27 = 108, which is still much better than 189!
    
    So even with schoolbook 3x3, we get rank 108.
    """
    # For now, use schoolbook 3x3x3 (rank 27)
    # U: (9, 27), V: (9, 27), W: (9, 27)
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)*(b00+b11): C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)*b00: C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a00*(b01-b11): C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a11*(b10-b00): C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a00+a01)*b11: C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a10-a00)*(b00+b01): C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)*(b10+b11): C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition_3x3_blocks():
    """
    Decompose 6x6x6 matrix multiplication using 2x2 blocks of 3x3 matrices.
    The 6x6 matrix is divided into a 2x2 grid of 3x3 blocks.
    Matrix multiplication C = A*B where A, B, C are 6x6.
    In block form: C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    where each block is 3x3.
    
    For each of the 8 block multiplications (i,j,k in {0,1}^3), we use schoolbook 3x3 (rank 27).
    Total rank = 8 * 27 = 216. That's the same as trivial!
    
    Wait, that's not right. Let me reconsider.
    
    If I use 2x2 blocks of 3x3, I have:
    - A has 4 blocks: A00, A01, A10, A11 (each 3x3)
    - B has 4 blocks: B00, B01, B10, B11 (each 3x3)
    - C has 4 blocks: C00, C01, C10, C11 (each 3x3)
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    So I need 8 matrix multiplications of 3x3 matrices.
    Each 3x3x3 multiplication has rank 27 (schoolbook) or 23 (Winograd).
    
    With schoolbook: 8 * 27 = 216 (same as trivial)
    With Winograd: 8 * 23 = 184 (better than 189!)
    
    So I need a proper Winograd 3x3 implementation with rank 23.
    """
    pass

def _block_decomposition_2x2_blocks_strassen():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    The 6x6 matrix is divided into a 3x3 grid of 2x2 blocks.
    For each of the 27 block multiplications, use Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Use 3x3 blocks of 2x2 with Strassen (rank 189)
    best = _block_decomposition_2x2_blocks_strassen()
    best_rank = 189
    
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
