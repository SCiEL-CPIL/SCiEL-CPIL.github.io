import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    # Strassen's 7-term algorithm for 2x2 matrix multiplication.
    # For matrices A (2x2) and B (2x2), C = A*B:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 2x2x8: group the 8 columns of B into 4 pairs.
    # Each pair forms a 2x2 sub-matrix, and Strassen applies to each.
    # Total rank: 4 * 7 = 28.
    #
    # Tensor convention:
    #   T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #   a = i*m + j  (i: row of A, j: col of A)
    #   b = j*p + k  (j: row of B, k: col of B)
    #   c = k*n + i  (k: col of C, i: row of C)  [trace convention: W[k*n+i, r]]
    #
    # Target: T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), else 0.

    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen products: each is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {(i,j): coeff} for the 2x2 A sub-matrix
    # B_coeffs: dict {(j,k_sub): coeff} for the 2x2 B sub-matrix
    # C_coeffs: dict {(i,k_sub): coeff} for contribution to C sub-matrix

    strassen_products = [
        # M1 = (A00+A11)*(B00+B11)
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)*B00
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00*(B01-B11)
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11*(B10-B00)
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)*B11
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)*(B00+B01)
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7 = (A01-A11)*(B10+B11)
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]

    # Build the rank-28 decomposition
    for q in range(p // 2):  # 4 groups of 2 columns
        k1 = 2 * q  # first column index in this group
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # U factor: linear combination of A[i,j]
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # V factor: linear combination of B[j, k] for k in {k1, k1+1}
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # W factor: contribution to C[i, k] for k in {k1, k1+1}
            # W index: k*n + i (trace convention)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff

    # Verify the decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kp = c // n
                    ip = c % n
                    
                    if i == ip and j == jp and k == kp:
                        if abs(T[a, b, c] - 1.0) > 1e-6:
                            return False
                    else:
                        if abs(T[a, b, c] - 0.0) > 1e-6:
                            return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-32 (should never be reached)
    rank = n * m * p  # 32
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
