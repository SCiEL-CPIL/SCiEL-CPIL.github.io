import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Rank-28 decomposition via Strassen's 7-product algorithm applied
    # to 4 independent 2x2 column blocks of B.
    #
    # For each group q (q=0,1,2,3), columns k1=2q, k2=2q+1:
    #   B_sub = [[B[0,k1], B[0,k2]],
    #            [B[1,k1], B[1,k2]]]
    #   C_sub = A * B_sub
    #
    # Strassen products (sub-matrix indices):
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # Reconstruction:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Each Strassen product defined as:
    #   A_coeffs: dict (i,j) -> coeff  (for the 2x2 A sub-block)
    #   B_coeffs: dict (j,k_sub) -> coeff  (for the 2x2 B sub-block)
    #   C_coeffs: dict (i,k_sub) -> coeff  (contribution to 2x2 C sub-block)
    
    strassen_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index 0..27
            
            # Fill U: A coefficients (index i*m + j)
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients (index j*p + k_global)
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions (index k_global*n + i)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verify the decomposition
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    if not np.allclose(T_computed, T_target, atol=1e-10):
        # Fallback to schoolbook (rank 32) if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
