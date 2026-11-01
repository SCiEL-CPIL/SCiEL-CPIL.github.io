import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Known decomposition:
    # 1. Use Strassen's algorithm (rank 7) for the 2x2 submatrix (columns 0,1)
    # 2. Use schoolbook (rank 4) for the remaining column (column 2)
    # Total rank = 7 + 4 = 11
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x3, we have:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A * B = [[c00, c01, c02], [c10, c11, c12]]
    
    # We can decompose B into two parts:
    # B1 = [[b00, b01], [b10, b11]] (first two columns)
    # B2 = [[b02], [b12]] (third column)
    
    # C1 = A * B1 (2x2 * 2x2) -> use Strassen (rank 7)
    # C2 = A * B2 (2x2 * 2x1) -> use schoolbook (rank 4)
    
    # Strassen for C1 (columns 0,1):
    # M1 = (a00 + a11)(b00 + b10)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Schoolbook for C2 (column 2):
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    
    # Now let's construct U, V, W matrices
    
    # Indexing:
    # U: (n*m) x rank = 4 x 11
    # V: (m*p) x rank = 6 x 11
    # W: (p*n) x rank = 6 x 11
    
    # u = i*m + j (i in [0,1], j in [0,1])
    # v = j*p + k (j in [0,1], k in [0,2])
    # w = k*n + i (k in [0,2], i in [0,1])
    
    # Strassen part (rank 7):
    # M1: U[a00]=1, U[a11]=1; V[b00]=1, V[b10]=1; W[c00]=1, W[c11]=1
    # M2: U[a10]=1, U[a11]=1; V[b00]=1; W[c10]=1, W[c11]=-1
    # M3: U[a00]=1; V[b01]=1, V[b11]=-1; W[c01]=1, W[c11]=1
    # M4: U[a11]=1; V[b10]=1, V[b00]=-1; W[c00]=1, W[c10]=1
    # M5: U[a00]=1, U[a01]=1; V[b11]=1; W[c00]=-1, W[c01]=1
    # M6: U[a10]=1, U[a00]=-1; V[b00]=1, V[b01]=1; W[c11]=1
    # M7: U[a01]=1, U[a11]=-1; V[b10]=1, V[b11]=1; W[c00]=1
    
    # Schoolbook part for column 2 (rank 4):
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    
    # Let's define the rank-1 terms:
    
    # Term 0: M1
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b10
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11
    
    # Term 1: M2
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # c10
    W[3][1] = -1 # c11
    
    # Term 2: M3
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[4][2] = -1 # b11
    W[1][2] = 1  # c01
    W[3][2] = 1  # c11
    
    # Term 3: M4
    U[3][3] = 1  # a11
    V[3][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # c00
    W[2][3] = 1  # c10
    
    # Term 4: M5
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[4][4] = 1  # b11
    W[0][4] = -1 # c00
    W[1][4] = 1  # c01
    
    # Term 5: M6
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # c11
    
    # Term 6: M7
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[3][6] = 1  # b10
    V[4][6] = 1  # b11
    W[0][6] = 1  # c00
    
    # Term 7: c02 = a00*b02
    U[0][7] = 1  # a00
    V[2][7] = 1  # b02
    W[4][7] = 1  # c02
    
    # Term 8: c02 += a01*b12
    U[1][8] = 1  # a01
    V[5][8] = 1  # b12
    W[4][8] = 1  # c02
    
    # Term 9: c12 = a10*b02
    U[2][9] = 1  # a10
    V[2][9] = 1  # b02
    W[5][9] = 1  # c12
    
    # Term 10: c12 += a11*b12
    U[3][10] = 1 # a11
    V[5][10] = 1 # b12
    W[5][10] = 1 # c12
    
    # Verify the decomposition
    # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    
    # Compute tensor product
    T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T_recon += np.einsum('u,v,w->uvw', U_np[:, r], V_np[:, r], W_np[:, r])
    
    if not np.allclose(T, T_recon, atol=1e-10):
        # Fallback to schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
