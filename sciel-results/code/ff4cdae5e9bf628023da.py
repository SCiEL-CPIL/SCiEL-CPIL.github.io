import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (B00 + B11)(A01 - A11)
    # M2 = (B00 + B01) A11
    # M3 = (A00 + A11)(B11 - B00)
    # M4 = A00 (B10 + B11)
    # M5 = (A10 - A00) B00
    # M6 = (A00 + A01) (B11 - B10)
    # M7 = (A10 - A01) (B00 + B01)
    
    # u[i, j] for A part, v[j, k] for B part, w[i, k] for C part
    # Indices for 2x2 block: i in {0,1}, j in {0,1}, k in {0,1} (local)
    
    # M1
    u1 = [[0, 1], [0, -1]]  # A01 - A11
    v1 = [[1, 0], [0, 1]]  # B00 + B11
    w1 = [[1, 0], [0, 1]]  # C00 += M1, C11 += M1
    
    # M2
    u2 = [[0, 0], [0, 1]]  # A11
    v2 = [[1, 1], [0, 0]]  # B00 + B01
    w2 = [[0, 0], [1, -1]] # C10 += M2, C11 -= M2
    
    # M3
    u3 = [[1, 0], [0, 1]]  # A00 + A11
    v3 = [[-1, 0], [0, 1]] # B11 - B00
    w3 = [[1, 0], [0, 1]]  # C00 += M3, C11 += M3
    
    # M4
    u4 = [[1, 0], [0, 0]]  # A00
    v4 = [[0, 0], [1, 1]]  # B10 + B11
    w4 = [[-1, 1], [0, 0]] # C00 -= M4, C01 += M4
    
    # M5
    u5 = [[-1, 0], [1, 0]] # A10 - A00
    v5 = [[1, 0], [0, 0]]  # B00
    w5 = [[0, 0], [1, 1]]  # C10 += M5, C11 += M5
    
    # M6
    u6 = [[1, 1], [0, 0]]  # A00 + A01
    v6 = [[0, 0], [0, 1]]  # B11 - B10 -> wait. B11 is (1,1), B10 is (1,0).
    # v6 should be B11 - B10.
    # B indices: 00, 01, 10, 11.
    # v6[j, k]. j=1, k=0 is B10. j=1, k=1 is B11.
    # So v6 = [[0, 0], [-1, 1]]
    v6 = [[0, 0], [-1, 1]]
    w6 = [[0, 1], [0, 0]]  # C01 += M6
    
    # M7
    u7 = [[0, -1], [1, 0]] # A10 - A01
    v7 = [[1, 1], [0, 0]]  # B00 + B01
    w7 = [[1, 0], [0, 0]]  # C00 += M7
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    col = 0
    for r in range(p // 2):
        # 7 Strassen terms
        for u, v, w in strassen_terms:
            # u is 2x2, v is 2x2, w is 2x2
            # Map to global indices
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = u[i][j]
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * r + k_local
                    V[j * p + k_global, col] = v[j][k_local]
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * r + k_local
                    W[k_global * n + i, col] = w[i][k_local]
            col += 1
            
        # 4 Schoolbook terms for j=2
        # Terms: A02 B2k, A12 B2k for k in {2r, 2r+1}
        # M8 = A02 B2,2r
        # M9 = A02 B2,2r+1
        # M10 = A12 B2,2r
        # M11 = A12 B2,2r+1
        
        # M8
        U[0 * m + 2, col] = 1
        V[2 * p + 2 * r, col] = 1
        W[2 * r * n + 0, col] = 1
        col += 1
        
        # M9
        U[0 * m + 2, col] = 1
        V[2 * p + 2 * r + 1, col] = 1
        W[(2 * r + 1) * n + 0, col] = 1
        col += 1
        
        # M10
        U[1 * m + 2, col] = 1
        V[2 * p + 2 * r, col] = 1
        W[2 * r * n + 1, col] = 1
        col += 1
        
        # M11
        U[1 * m + 2, col] = 1
        V[2 * p + 2 * r + 1, col] = 1
        W[(2 * r + 1) * n + 1, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
