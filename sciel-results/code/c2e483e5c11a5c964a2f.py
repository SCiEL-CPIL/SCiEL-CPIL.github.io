import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build Strassen 2x2x2 decomposition (rank 7)
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1  # C00
    W_s[3,0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[2,1] = 1  # C10
    
    # M3 = A00(B01-B11)
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[1,2] = 1  # C01
    W_s[2,2] = 1  # C10
    
    # M4 = A11(B10-B00)
    U_s[3,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[0,3] = -1  # C00
    W_s[1,3] = 1   # C01
    
    # M5 = (A00+A01)B11
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[0,4] = 1  # C00
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2,5] = 1; U_s[0,5] = -1
    V_s[0,5] = 1; V_s[1,5] = 1
    W_s[3,5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1,6] = 1; U_s[3,6] = -1
    V_s[2,6] = 1; V_s[3,6] = 1
    W_s[0,6] = -1  # C00
    W_s[3,6] = -1  # C11
    
    # Verify 2x2x2 Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(7):
                    val += U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r]
                if abs(val - 1.0) > 1e-10:
                    return _schoolbook(n, m, p)
    
    # Build 4x4x9: use 2x2x4 blocks (Strassen for first 2 cols + schoolbook for last 2)
    # Partition: A into 2x2 blocks, B into 2x4 blocks, C into 4x2 blocks
    # 4 blocks of 2x2x4, each with rank 7+2=9, plus 4 schoolbook for col 8
    rank = 4*9 + 4
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Strassen part for columns 2*bk, 2*bk+1
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[(2*bi+i)*m + (2*bj+j), col] = U_s[i*2+j, r]
                    for j in range(2):
                        for k in range(2):
                            V[(2*bj+j)*p + (2*bk+k), col] = V_s[j*2+k, r]
                    for k in range(2):
                        for i in range(2):
                            W[(2*bk+k)*n + (2*bi+i), col] = W_s[k*2+i, r]
                    col += 1
                # Schoolbook part for columns 2*bk+2, 2*bk+3
                for k in range(2):
                    for i in range(2):
                        for j in range(2):
                            U[(2*bi+i)*m + (2*bj+j), col] = 1
                            V[(2*bj+j)*p + (2*bk+2+k), col] = 1
                            W[(2*bk+2+k)*n + (2*bi+i), col] = 1
                            col += 1
    
    # Schoolbook for column 8
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p + (p-1), col] = 1
            W[(p-1)*n + i, col] = 1
            col += 1
    
    # Verify full decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return _schoolbook(n, m, p)
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-10:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    return {"U": clean_matrix(U.tolist()), "V": clean_matrix(V.tolist()), "W": clean_matrix(W.tolist())}

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
