import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Rank 28 decomposition using Strassen's algorithm for 4 groups of 2 columns.
    # Each group uses 7 Strassen products, total 4*7 = 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrix multiplication.
    # For A (2x2) and B_sub (2x2), C_sub = A * B_sub:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    # Outputs:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {'A': {(0,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0, (1,1): 1.0}, 
         'C': {(0,0): 1.0, (1,1): 1.0}},
        # M2 = (A10+A11)B00
        {'A': {(1,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0}, 
         'C': {(1,0): 1.0, (1,1): -1.0}},
        # M3 = A00(B01-B11)
        {'A': {(0,0): 1.0}, 
         'B': {(0,1): 1.0, (1,1): -1.0}, 
         'C': {(0,1): 1.0, (1,1): 1.0}},
        # M4 = A11(B10-B00)
        {'A': {(1,1): 1.0}, 
         'B': {(1,0): 1.0, (0,0): -1.0}, 
         'C': {(0,0): 1.0, (1,0): 1.0}},
        # M5 = (A00+A01)B11
        {'A': {(0,0): 1.0, (0,1): 1.0}, 
         'B': {(1,1): 1.0}, 
         'C': {(0,0): -1.0, (0,1): 1.0}},
        # M6 = (A10-A00)(B00+B01)
        {'A': {(1,0): 1.0, (0,0): -1.0}, 
         'B': {(0,0): 1.0, (0,1): 1.0}, 
         'C': {(1,1): 1.0}},
        # M7 = (A01-A11)(B10+B11)
        {'A': {(0,1): 1.0, (1,1): -1.0}, 
         'B': {(1,0): 1.0, (1,1): 1.0}, 
         'C': {(0,0): 1.0}},
    ]
    
    # For each group of 2 columns (q = 0,1,2,3), apply Strassen.
    # Group q uses columns k1=2q, k2=2q+1.
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, term in enumerate(strassen_terms):
            t = q * 7 + r  # term index
            
            # Fill U: indexed by i*m + j
            for (i, j), coeff in term['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: indexed by j*p + k, where k = k1 + k_sub
            for (j, k_sub), coeff in term['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: indexed by k*n + i, where k = k1 + k_sub
            for (i, k_sub), coeff in term['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Verify the decomposition
    if time.time() - start_time < deadline - 5:
        # Correct verification:
        # The tensor T[i,j,k] for matrix multiplication is 1 for all i,j,k.
        # A rank-r decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We check this identity for all i, j, k.
        
        all_correct = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the sum for this specific tensor entry
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-10:
                        all_correct = False
                        break
                if not all_correct:
                    break
            if not all_correct:
                break
        
        if not all_correct:
            # Fallback: use schoolbook rank 32
            rank = 32
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t = i * m * p + j * p + k
                        U[i * m + j, t] = 1.0
                        V[j * p + k, t] = 1.0
                        W[k * n + i, t] = 1.0
    
    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
