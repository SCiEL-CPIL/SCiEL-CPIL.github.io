import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # U: (n*m, rank) = (6, 22)
    # V: (m*p, rank) = (12, 22)
    # W: (p*n, rank) = (8, 22)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: list of (index, coeff)
    # v_idx: list of (index, coeff)
    # w_idx: list of (index, coeff)
    def add_term(col, u_idx, v_idx, w_idx):
        for idx, coeff in u_idx:
            U[idx][col] += coeff
        for idx, coeff in v_idx:
            V[idx][col] += coeff
        for idx, coeff in w_idx:
            W[idx][col] += coeff
            
    # Block 1: Columns 0,1 of C
    # A indices: 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2)
    # B indices: 0:(0,0) 1:(0,1) 2:(0,2) 3:(0,3) 4:(1,0) 5:(1,1) 6:(1,2) 7:(1,3) 8:(2,0) 9:(2,1) 10:(2,2) 11:(2,3)
    # C indices: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1) 4:(0,2) 5:(0,3) 6:(1,2) 7:(1,3)
    
    # Strassen for A_L (2x2) * B_L (2x2)
    # A_L uses A[0], A[1], A[3], A[4]
    # B_L uses B[0], B[1], B[4], B[5]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    add_term(0, [(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)])
    
    # Term 1: M2 = (A10+A11)B00
    add_term(1, [(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)])
    
    # Term 2: M3 = A00(B01-B11)
    add_term(2, [(0, 1)], [(1, 1), (5, -1)], [(1, 1), (3, 1)])
    
    # Term 3: M4 = A11(B10-B00)
    add_term(3, [(4, 1)], [(4, 1), (0, -1)], [(0, 1), (2, 1)])
    
    # Term 4: M5 = (A00+A01)B11
    add_term(4, [(0, 1), (1, 1)], [(5, 1)], [(0, -1), (1, 1)])
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    add_term(5, [(3, 1), (0, -1)], [(0, 1), (4, 1)], [(3, 1)])
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    add_term(6, [(1, 1), (4, -1)], [(1, 1), (5, 1)], [(0, 1)])
    
    # Rank 4 for A_R (2x1) * B_R (1x2)
    # A_R uses A[2], A[5]
    # B_R uses B[8], B[9]
    
    # Term 7: N1 = A02 B20
    add_term(7, [(2, 1)], [(8, 1)], [(0, 1)])
    
    # Term 8: N2 = A02 B21
    add_term(8, [(2, 1)], [(9, 1)], [(1, 1)])
    
    # Term 9: N3 = A12 B20
    add_term(9, [(5, 1)], [(8, 1)], [(2, 1)])
    
    # Term 10: N4 = A12 B21
    add_term(10, [(5, 1)], [(9, 1)], [(3, 1)])
    
    # Block 2: Columns 2,3 of C
    # B_L' uses B[2], B[3], B[6], B[7]
    # B_R' uses B[10], B[11]
    # C indices 4,5,6,7
    
    # Term 11: M1' = (A00+A11)(B02+B12)
    add_term(11, [(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)])
    
    # Term 12: M2' = (A10+A11)B02
    add_term(12, [(3, 1), (4, 1)], [(2, 1)], [(6, 1), (7, -1)])
    
    # Term 13: M3' = A00(B03-B12)
    add_term(13, [(0, 1)], [(3, 1), (7, -1)], [(5, 1), (7, 1)])
    
    # Term 14: M4' = A11(B12-B02)
    add_term(14, [(4, 1)], [(6, 1), (2, -1)], [(4, 1), (6, 1)])
    
    # Term 15: M5' = (A00+A01)B12
    add_term(15, [(0, 1), (1, 1)], [(7, 1)], [(4, -1), (5, 1)])
    
    # Term 16: M6' = (A10-A00)(B02+B12)
    add_term(16, [(3, 1), (0, -1)], [(2, 1), (6, 1)], [(7, 1)])
    
    # Term 17: M7' = (A01-A11)(B03+B12)
    add_term(17, [(1, 1), (4, -1)], [(3, 1), (7, 1)], [(4, 1)])
    
    # Term 18: N1' = A02 B22
    add_term(18, [(2, 1)], [(10, 1)], [(4, 1)])
    
    # Term 19: N2' = A02 B23
    add_term(19, [(2, 1)], [(11, 1)], [(5, 1)])
    
    # Term 20: N3' = A12 B22
    add_term(20, [(5, 1)], [(10, 1)], [(6, 1)])
    
    # Term 21: N4' = A12 B23
    add_term(21, [(5, 1)], [(11, 1)], [(7, 1)])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
