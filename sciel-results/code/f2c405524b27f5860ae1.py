import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Strassen products for 2x2 matrix multiplication
    strassen_products = [
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    def build_decomposition(terms_to_include):
        """Build U, V, W for the given set of term indices."""
        rank = len(terms_to_include)
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        for idx, t in enumerate(terms_to_include):
            q, r = divmod(t, 7)
            k1 = 2 * q
            prod = strassen_products[r]
            
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, idx] = coeff
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, idx] = coeff
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, idx] = coeff
        
        return U, V, W
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Try all 28 terms first
    all_terms = list(range(28))
    U, V, W = build_decomposition(all_terms)
    
    # Verify
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.array_equal(T_computed, T_target):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to remove one term at a time
    if time.time() < deadline - 20:
        for t_to_remove in range(28):
            terms = [t for t in range(28) if t != t_to_remove]
            U_test, V_test, W_test = build_decomposition(terms)
            T_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            if np.array_equal(T_test, T_target):
                # Found a rank-27 solution!
                U, V, W = U_test, V_test, W_test
                break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
