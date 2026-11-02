"""Matrix multiplication 5x5x5 using block structure: Laderman(3x3) + Strassen(2x2) + schoolbook cross terms."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-82 bilinear algorithm for 5x5 matrix multiplication.
    
    Strategy:
    - Laderman's algorithm (rank 15) for the 3x3 subblock C[0:3, 0:3]
    - Strassen's algorithm (rank 7) for the 2x2 subblock C[3:5, 3:5]
    - Schoolbook products for cross terms: C[0:3, 3:5] and C[3:5, 0:3]
      Cross terms 1 (C[i,j] for i in {0,1,2}, j in {3,4}): 3*5*2 = 30 products
      Cross terms 2 (C[i,j] for i in {3,4}, j in {0,1,2}): 2*5*3 = 30 products
    Total: 15 + 7 + 30 + 30 = 82
    """
    n, m, p = DIMENSIONS
    rank = 82
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # ============================================================
    # LADERMAN'S ALGORITHM for 3x3 matrix multiplication (rank 15)
    # Subproblem: C[0:3, 0:3] = A[0:3, 0:3] * B[0:3, 0:3]
    # ============================================================
    
    # Local indices: A_sub[i,k] = A[i,k] for i,k in {0,1,2}
    #                B_sub[k,j] = B[k,j] for k,j in {0,1,2}
    #                C_sub[i,j] = C[i,j] for i,j in {0,1,2}
    
    def A_idx(i, k):
        return i * m + k
    
    def B_idx(k, j):
        return k * p + j
    
    def C_idx(i, j):
        return j * n + i
    
    # Laderman's 15 products
    # m1 = A[0,0]*B[0,0]
    U[A_idx(0,0)][col] = 1; V[B_idx(0,0)][col] = 1; col += 1
    m1 = col - 1
    
    # m2 = (A[0,0]+A[0,1])*B[1,0]
    U[A_idx(0,0)][col] = 1; V[B_idx(1,0)][col] = 1
    U[A_idx(0,1)][col] = 1; V[B_idx(1,0)][col] += 1; col += 1
    m2 = col - 1
    
    # m3 = (A[1,0]+A[1,1])*B[0,0]
    U[A_idx(1,0)][col] = 1; V[B_idx(0,0)][col] = 1
    U[A_idx(1,1)][col] = 1; V[B_idx(0,0)][col] += 1; col += 1
    m3 = col - 1
    
    # m4 = A[2,2]*B[2,2]
    U[A_idx(2,2)][col] = 1; V[B_idx(2,2)][col] = 1; col += 1
    m4 = col - 1
    
    # m5 = (A[0,2]+A[2,2])*B[2,0]
    U[A_idx(0,2)][col] = 1; V[B_idx(2,0)][col] = 1
    U[A_idx(2,2)][col] = 1; V[B_idx(2,0)][col] += 1; col += 1
    m5 = col - 1
    
    # m6 = (A[2,0]+A[2,2])*B[0,2]
    U[A_idx(2,0)][col] = 1; V[B_idx(0,2)][col] = 1
    U[A_idx(2,2)][col] = 1; V[B_idx(0,2)][col] += 1; col += 1
    m6 = col - 1
    
    # m7 = (A[1,1]-A[1,0])*(B[1,0]+B[2,0])
    U[A_idx(1,1)][col] = 1; V[B_idx(1,0)][col] = 1
    U[A_idx(1,0)][col] = -1; V[B_idx(1,0)][col] += 1
    U[A_idx(1,1)][col] += 1; V[B_idx(2,0)][col] = 1
    U[A_idx(1,0)][col] -= 1; V[B_idx(2,0)][col] += 1; col += 1
    m7 = col - 1
    
    # m8 = (A[0,1]-A[0,0])*(B[0,0]+B[1,0])
    U[A_idx(0,1)][col] = 1; V[B_idx(0,0)][col] = 1
    U[A_idx(0,0)][col] = -1; V[B_idx(0,0)][col] += 1
    U[A_idx(0,1)][col] += 1; V[B_idx(1,0)][col] = 1
    U[A_idx(0,0)][col] -= 1; V[B_idx(1,0)][col] += 1; col += 1
    m8 = col - 1
    
    # m9 = (A[1,1]+A[2,2])*(B[1,1]+B[2,2])
    U[A_idx(1,1)][col] = 1; V[B_idx(1,1)][col] = 1
    U[A_idx(2,2)][col] = 1; V[B_idx(1,1)][col] += 1
    U[A_idx(1,1)][col] += 1; V[B_idx(2,2)][col] = 1
    U[A_idx(2,2)][col] += 1; V[B_idx(2,2)][col] += 1; col += 1
    m9 = col - 1
    
    # m10 = (A[0,1]+A[1,1])*(B[0,1]+B[1,1])
    U[A_idx(0,1)][col] = 1; V[B_idx(0,1)][col] = 1
    U[A_idx(1,1)][col] = 1; V[B_idx(0,1)][col] += 1
    U[A_idx(0,1)][col] += 1; V[B_idx(1,1)][col] = 1
    U[A_idx(1,1)][col] += 1; V[B_idx(1,1)][col] += 1; col += 1
    m10 = col - 1
    
    # m11 = (A[1,1]+A[2,1])*(B[1,2]+B[2,2])
    U[A_idx(1,1)][col] = 1; V[B_idx(1,2)][col] = 1
    U[A_idx(2,1)][col] = 1; V[B_idx(1,2)][col] += 1
    U[A_idx(1,1)][col] += 1; V[B_idx(2,2)][col] = 1
    U[A_idx(2,1)][col] += 1; V[B_idx(2,2)][col] += 1; col += 1
    m11 = col - 1
    
    # m12 = (A[0,2]+A[1,2])*(B[0,2]+B[1,2])
    U[A_idx(0,2)][col] = 1; V[B_idx(0,2)][col] = 1
    U[A_idx(1,2)][col] = 1; V[B_idx(0,2)][col] += 1
    U[A_idx(0,2)][col] += 1; V[B_idx(1,2)][col] = 1
    U[A_idx(1,2)][col] += 1; V[B_idx(1,2)][col] += 1; col += 1
    m12 = col - 1
    
    # m13 = (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
    U[A_idx(0,0)][col] = 1; V[B_idx(0,0)][col] = 1
    U[A_idx(2,2)][col] = 1; V[B_idx(0,0)][col] += 1
    U[A_idx(0,0)][col] += 1; V[B_idx(2,2)][col] = 1
    U[A_idx(2,2)][col] += 1; V[B_idx(2,2)][col] += 1; col += 1
    m13 = col - 1
    
    # m14 = (A[0,2]+A[1,2]+A[2,2])*(B[0,0]+B[1,1]+B[2,0])
    U[A_idx(0,2)][col] = 1; V[B_idx(0,0)][col] = 1
    U[A_idx(1,2)][col] = 1; V[B_idx(0,0)][col] += 1
    U[A_idx(2,2)][col] = 1; V[B_idx(0,0)][col] += 1
    U[A_idx(0,2)][col] += 1; V[B_idx(1,1)][col] = 1
    U[A_idx(1,2)][col] += 1; V[B_idx(1,1)][col] += 1
    U[A_idx(2,2)][col] += 1; V[B_idx(1,1)][col] += 1
    U[A_idx(0,2)][col] += 1; V[B_idx(2,0)][col] = 1
    U[A_idx(1,2)][col] += 1; V[B_idx(2,0)][col] += 1
    U[A_idx(2,2)][col] += 1; V[B_idx(2,0)][col] += 1; col += 1
    m14 = col - 1
    
    # m15 = (A[0,0]+A[1,1]+A[2,2])*(B[0,2]+B[1,2]+B[2,0])
    U[A_idx(0,0)][col] = 1; V[B_idx(0,2)][col] = 1
    U[A_idx(1,1)][col] = 1; V[B_idx(0,2)][col] += 1
    U[A_idx(2,2)][col] = 1; V[B_idx(0,2)][col] += 1
    U[A_idx(0,0)][col] += 1; V[B_idx(1,2)][col] = 1
    U[A_idx(1,1)][col] += 1; V[B_idx(1,2)][col] += 1
    U[A_idx(2,2)][col] += 1; V[B_idx(1,2)][col] += 1
    U[A_idx(0,0)][col] += 1; V[B_idx(2,0)][col] = 1
    U[A_idx(1,1)][col] += 1; V[B_idx(2,0)][col] += 1
    U[A_idx(2,2)][col] += 1; V[B_idx(2,0)][col] += 1; col += 1
    m15 = col - 1
    
    # Laderman output combinations:
    # C[0,0] = m1 + m7 - m8 - m9 + m13
    W[C_idx(0,0)][m1] += 1; W[C_idx(0,0)][m7] += 1; W[C_idx(0,0)][m8] -= 1; W[C_idx(0,0)][m9] -= 1; W[C_idx(0,0)][m13] += 1
    # C[0,1] = m8 + m10 - m13 - m15
    W[C_idx(0,1)][m8] += 1; W[C_idx(0,1)][m10] += 1; W[C_idx(0,1)][m13] -= 1; W[C_idx(0,1)][m15] -= 1
    # C[0,2] = m1 + m6 - m12 - m14
    W[C_idx(0,2)][m1] += 1; W[C_idx(0,2)][m6] += 1; W[C_idx(0,2)][m12] -= 1; W[C_idx(0,2)][m14] -= 1
    # C[1,0] = m3 - m7 + m13 - m15
    W[C_idx(1,0)][m3] += 1; W[C_idx(1,0)][m7] -= 1; W[C_idx(1,0)][m13] += 1; W[C_idx(1,0)][m15] -= 1
    # C[1,1] = m9 - m10 + m11 + m14 - m15
    W[C_idx(1,1)][m9] += 1; W[C_idx(1,1)][m10] -= 1; W[C_idx(1,1)][m11] += 1; W[C_idx(1,1)][m14] += 1; W[C_idx(1,1)][m15] -= 1
    # C[1,2] = m5 - m11 + m12 + m14 - m13
    W[C_idx(1,2)][m5] += 1; W[C_idx(1,2)][m11] -= 1; W[C_idx(1,2)][m12] += 1; W[C_idx(1,2)][m14] += 1; W[C_idx(1,2)][m13] -= 1
    # C[2,0] = m2 - m3 + m13 - m14
    W[C_idx(2,0)][m2] += 1; W[C_idx(2,0)][m3] -= 1; W[C_idx(2,0)][m13] += 1; W[C_idx(2,0)][m14] -= 1
    # C[2,1] = m7 - m9 + m11 + m15
    W[C_idx(2,1)][m7] += 1; W[C_idx(2,1)][m9] -= 1; W[C_idx(2,1)][m11] += 1; W[C_idx(2,1)][m15] += 1
    # C[2,2] = m4 - m5 + m6 + m10 - m14
    W[C_idx(2,2)][m4] += 1; W[C_idx(2,2)][m5] -= 1; W[C_idx(2,2)][m6] += 1; W[C_idx(2,2)][m10] += 1; W[C_idx(2,2)][m14] -= 1
    
    # col should be 15
    
    # ============================================================
    # STRASSEN'S ALGORITHM for 2x2 matrix multiplication (rank 7)
    # Subproblem: C[3:5, 3:5] = A[3:5, 3:5] * B[3:5, 3:5]
    # ============================================================
    
    # Local: A_s[i,k] = A[3+i, 3+k], B_s[k,j] = B[3+k, 3+j], C_s[i,j] = C[3+i, 3+j]
    
    def As_idx(i, k):
        return (3 + i) * m + (3 + k)
    
    def Bs_idx(k, j):
        return (3 + k) * p + (3 + j)
    
    def Cs_idx(i, j):
        return (3 + j) * n + (3 + i)
    
    # M1 = (A[3,3]+A[4,4])*(B[3,3]+B[4,4])
    U[As_idx(0,0)][col] = 1; V[Bs_idx(0,0)][col] = 1
    U[As_idx(1,1)][col] = 1; V[Bs_idx(0,0)][col] += 1
    U[As_idx(0,0)][col] += 1; V[Bs_idx(1,1)][col] = 1
    U[As_idx(1,1)][col] += 1; V[Bs_idx(1,1)][col] += 1; col += 1
    s1 = col - 1
    
    # M2 = (A[4,3]+A[4,4])*B[3,3]
    U[As_idx(1,0)][col] = 1; V[Bs_idx(0,0)][col] = 1
    U[As_idx(1,1)][col] = 1; V[Bs_idx(0,0)][col] += 1; col += 1
    s2 = col - 1
    
    # M3 = A[3,3]*(B[4,3]-B[4,4])
    U[As_idx(0,0)][col] = 1; V[Bs_idx(1,0)][col] = 1
    U[As_idx(0,0)][col] += 1; V[Bs_idx(1,1)][col] = -1; col += 1
    s3 = col - 1
    
    # M4 = A[4,4]*(B[3,4]-B[4,3])
    U[As_idx(1,1)][col] = 1; V[Bs_idx(0,1)][col] = 1
    U[As_idx(1,1)][col] += 1; V[Bs_idx(1,0)][col] = -1; col += 1
    s4 = col - 1
    
    # M5 = (A[3,3]+A[3,4])*B[4,4]
    U[As_idx(0,0)][col] = 1; V[Bs_idx(1,1)][col] = 1
    U[As_idx(0,1)][col] = 1; V[Bs_idx(1,1)][col] += 1; col += 1
    s5 = col - 1
    
    # M6 = (A[4,3]-A[3,3])*(B[3,3]+B[3,4])
    U[As_idx(1,0)][col] = 1; V[Bs_idx(0,0)][col] = 1
    U[As_idx(0,0)][col] = -1; V[Bs_idx(0,0)][col] += 1
    U[As_idx(1,0)][col] += 1; V[Bs_idx(0,1)][col] = 1
    U[As_idx(0,0)][col] -= 1; V[Bs_idx(0,1)][col] += 1; col += 1
    s6 = col - 1
    
    # M7 = (A[3,4]-A[4,4])*(B[3,4]+B[4,4])
    U[As_idx(0,1)][col] = 1; V[Bs_idx(0,1)][col] = 1
    U[As_idx(1,1)][col] = -1; V[Bs_idx(0,1)][col] += 1
    U[As_idx(0,1)][col] += 1; V[Bs_idx(1,1)][col] = 1
    U[As_idx(1,1)][col] -= 1; V[Bs_idx(1,1)][col] += 1; col += 1
    s7 = col - 1
    
    # Strassen output:
    # C[3,3] = M1 + M4 - M5 + M7
    W[Cs_idx(0,0)][s1] += 1; W[Cs_idx(0,0)][s4] += 1; W[Cs_idx(0,0)][s5] -= 1; W[Cs_idx(0,0)][s7] += 1
    # C[3,4] = M3 + M5
    W[Cs_idx(0,1)][s3] += 1; W[Cs_idx(0,1)][s5] += 1
    # C[4,3] = M2 + M4
    W[Cs_idx(1,0)][s2] += 1; W[Cs_idx(1,0)][s4] += 1
    # C[4,4] = M1 - M2 + M3 + M6
    W[Cs_idx(1,1)][s1] += 1; W[Cs_idx(1,1)][s2] -= 1; W[Cs_idx(1,1)][s3] += 1; W[Cs_idx(1,1)][s6] += 1
    
    # col should be 22
    
    # ============================================================
    # CROSS TERMS: C[i,j] for i in {0,1,2}, j in {3,4}
    # C[i,j] = sum_k A[i,k]*B[k,j] for k=0..4
    # Schoolbook: 3 rows * 2 cols * 5 products each = 30 products
    # ============================================================
    
    for i in range(3):
        for j in range(3, 5):
            for k in range(5):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # col should be 52
    
    # ============================================================
    # CROSS TERMS: C[i,j] for i in {3,4}, j in {0,1,2}
    # C[i,j] = sum_k A[i,k]*B[k,j] for k=0..4
    # Schoolbook: 2 rows * 3 cols * 5 products each = 30 products
    # ============================================================
    
    for i in range(3, 5):
        for j in range(3):
            for k in range(5):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # col should be 82
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
