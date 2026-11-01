"""Rewritten program for 2x3x5 matrix multiplication rank minimization.
Uses a Strassen-like block decomposition to reduce rank from 30 to 28."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-28 exact tensor decomposition for 2x3x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w):
        nonlocal col
        for idx, coeff in zip(u_indices, coeffs_u):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, coeffs_v):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, coeffs_w):
            W[idx][col] += coeff
        col += 1
        
    # Helper to add Strassen 2x2x2 block
    def add_strassen_222(a_idx, b_idx, c_idx):
        # a_idx: [A00, A01, A10, A11]
        # b_idx: [B00, B01, B10, B11]
        # c_idx: [C00, C01, C10, C11]
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term([a_idx[0], a_idx[3]], [b_idx[0], b_idx[3]], [c_idx[0], c_idx[1], c_idx[2], c_idx[3]], 
                 [1, 1], [1, 1], [1, 0, 0, 1])
        
        # M2 = (A10 + A11)B00
        add_term([a_idx[2], a_idx[3]], [b_idx[0]], [c_idx[2], c_idx[3]], 
                 [1, 1], [1], [1, -1])
        
        # M3 = A00(B01 - B11)
        add_term([a_idx[0]], [b_idx[1], b_idx[3]], [c_idx[1], c_idx[3]], 
                 [1], [1, -1], [1, 1])
        
        # M4 = A11(B10 - B00)
        add_term([a_idx[3]], [b_idx[2], b_idx[0]], [c_idx[0], c_idx[2]], 
                 [1], [1, -1], [1, 1])
        
        # M5 = (A00 + A01)B11
        add_term([a_idx[0], a_idx[1]], [b_idx[3]], [c_idx[0], c_idx[1]], 
                 [1, 1], [1], [-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term([a_idx[2], a_idx[0]], [b_idx[0], b_idx[1]], [c_idx[3]], 
                 [1, -1], [1, 1], [1])
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term([a_idx[1], a_idx[3]], [b_idx[2], b_idx[3]], [c_idx[0]], 
                 [1, -1], [1, 1], [1])
        
    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_221(a_idx, b_idx, c_idx):
        # a_idx: [A00, A01, A10, A11]
        # b_idx: [B00, B10]
        # c_idx: [C00, C10]
        
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Term 1: A00 * B00 -> C00
        add_term([a_idx[0]], [b_idx[0]], [c_idx[0]], [1], [1], [1])
        
        # Term 2: A01 * B10 -> C00
        add_term([a_idx[1]], [b_idx[1]], [c_idx[0]], [1], [1], [1])
        
        # Term 3: A10 * B00 -> C10
        add_term([a_idx[2]], [b_idx[0]], [c_idx[1]], [1], [1], [1])
        
        # Term 4: A11 * B10 -> C10
        add_term([a_idx[3]], [b_idx[1]], [c_idx[1]], [1], [1], [1])

    # Helper to add Schoolbook 2x1x5 block
    def add_schoolbook_215(a_idx, b_idx, c_idx):
        # a_idx: [A02, A12]
        # b_idx: [B20, B21, B22, B23, B24]
        # c_idx: [C00, C01, C02, C03, C04, C10, C11, C12, C13, C14]
        
        for i in range(2):
            for k in range(5):
                # C[i, k] = A[i, 2] * B[2, k]
                # c_idx index: i*5 + k? No, W[k*n+i] = W[k*2+i].
                # So for row i, col k, index is k*2 + i.
                c_idx_local = k * 2 + i
                add_term([a_idx[i]], [b_idx[k]], [c_idx[c_idx_local]], [1], [1], [1])

    # 1. Strassen for A[:, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1 (row 0), 3,4 (row 1) -> [0, 1, 3, 4]
    # B indices: 0,1 (row 0), 5,6 (row 1) -> [0, 1, 5, 6]
    # C indices: k=0,i=0->0; k=1,i=0->1; k=0,i=1->2; k=1,i=1->3 -> [0, 1, 2, 3]
    add_strassen_222([0, 1, 3, 4], [0, 1, 5, 6], [0, 1, 2, 3])
    
    # 2. Strassen for A[:, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: [0, 1, 3, 4]
    # B indices: 2,3 (row 0), 7,8 (row 1) -> [2, 3, 7, 8]
    # C indices: k=2,i=0->4; k=3,i=0->5; k=2,i=1->6; k=3,i=1->7 -> [4, 5, 6, 7]
    add_strassen_222([0, 1, 3, 4], [2, 3, 7, 8], [4, 5, 6, 7])
    
    # 3. Schoolbook for A[:, 0:2] @ B[0:2, 4:5] -> C[0:2, 4:5]
    # A indices: [0, 1, 3, 4]
    # B indices: 4 (row 0), 9 (row 1) -> [4, 9]
    # C indices: k=4,i=0->8; k=4,i=1->9 -> [8, 9]
    add_schoolbook_221([0, 1, 3, 4], [4, 9], [8, 9])
    
    # 4. Schoolbook for A[:, 2:3] @ B[2:3, 0:5] -> C[0:2, 0:5]
    # A indices: 2 (row 0), 5 (row 1) -> [2, 5]
    # B indices: 10, 11, 12, 13, 14 -> [10, 11, 12, 13, 14]
    # C indices: 0..9
    add_schoolbook_215([2, 5], [10, 11, 12, 13, 14], list(range(10)))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
