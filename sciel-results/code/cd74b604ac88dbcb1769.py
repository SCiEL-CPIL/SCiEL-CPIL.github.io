import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep safety margin
    
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term to the decomposition
    # u_idx: index in A (i*m + j)
    # v_idx: index in B (j*p + k)
    # w_idx: index in C (k*n + i)
    # coeff: coefficient for this term in the rank-1 tensor
    def add_term(u_indices, v_indices, w_indices, coeff, col_idx):
        for u_idx in u_indices:
            U[u_idx, col_idx] += coeff
        for v_idx in v_indices:
            V[v_idx, col_idx] += coeff
        for w_idx in w_indices:
            W[w_idx, col_idx] += coeff

    # Indices mapping:
    # A (n x m): idx = i*m + j
    # B (m x p): idx = j*p + k
    # C (p x n): idx = k*n + i (Trace convention)
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1: (A00 + A11)(B00 + B11)
    # U: A00(0), A11(4); V: B00(0), B11(4); W: C00(0), C11(3)
    add_term([0, 4], [0, 4], [0, 3], 1, 0)
    
    # M2: (A10 + A11)B00
    # U: A10(3), A11(4); V: B00(0); W: C10(1), -C11(-3)
    # Note: W contribution is split. The term is M2 * (e_C10 - e_C11)
    # So we add coeff 1 to W[1] and coeff -1 to W[3]
    # But the function add_term adds same coeff to all.
    # We need to handle mixed coefficients in W or U/V.
    # Let's refine add_term to take lists of (index, coeff)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term_rich(u_list, v_list, w_list, col_idx):
        for idx, coeff in u_list:
            U[idx, col_idx] += coeff
        for idx, coeff in v_list:
            V[idx, col_idx] += coeff
        for idx, coeff in w_list:
            W[idx, col_idx] += coeff

    # M1: (A00 + A11)(B00 + B11) -> C00 + C11
    add_term_rich([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    
    # M2: (A10 + A11)B00 -> C10 - C11
    add_term_rich([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # M3: A00(B01 - B11) -> C10 + C11
    add_term_rich([(0, 1)], [(1, 1), (4, -1)], [(1, 1), (3, 1)], 2)
    
    # M4: A11(B10 - B00) -> C00 + C10
    add_term_rich([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    
    # M5: (A00 + A01)B11 -> -C00 + C01
    add_term_rich([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    
    # M6: (A10 - A00)(B00 + B01) -> C01
    add_term_rich([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1)], 5)
    
    # M7: (A01 - A11)(B10 + B11) -> C00
    add_term_rich([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)
    
    # Block 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # i in {0,1}, j=2, k in {0,1}
    
    # r=7: A02 B20 -> C00
    add_term_rich([(2, 1)], [(6, 1)], [(0, 1)], 7)
    
    # r=8: A02 B21 -> C01
    add_term_rich([(2, 1)], [(7, 1)], [(2, 1)], 8)
    
    # r=9: A12 B20 -> C10
    add_term_rich([(5, 1)], [(6, 1)], [(1, 1)], 9)
    
    # r=10: A12 B21 -> C11
    add_term_rich([(5, 1)], [(7, 1)], [(3, 1)], 10)
    
    # Block 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # i in {0,1}, j in {0,1,2}, k=2
    
    # r=11: A00 B02 -> C02
    add_term_rich([(0, 1)], [(2, 1)], [(4, 1)], 11)
    
    # r=12: A01 B12 -> C02
    add_term_rich([(1, 1)], [(5, 1)], [(4, 1)], 12)
    
    # r=13: A02 B22 -> C02
    add_term_rich([(2, 1)], [(8, 1)], [(4, 1)], 13)
    
    # r=14: A10 B02 -> C12
    add_term_rich([(3, 1)], [(2, 1)], [(5, 1)], 14)
    
    # r=15: A11 B12 -> C12
    add_term_rich([(4, 1)], [(5, 1)], [(5, 1)], 15)
    
    # r=16: A12 B22 -> C12
    add_term_rich([(5, 1)], [(8, 1)], [(5, 1)], 16)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        pass # Return immediately as construction is static
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
