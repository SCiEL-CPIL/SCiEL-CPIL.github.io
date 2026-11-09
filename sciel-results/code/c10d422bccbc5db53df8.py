"""
Optimized approach: Split 2x3 * 3x6 into 2x2 * 2x6 (Strassen) and 2x1 * 1x6 (Schoolbook).
Rank 33 vs Schoolbook 36.
"""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Global dimensions
    n_A = n * m  # 6
    n_B = m * p  # 18
    n_C = p * n  # 12
    
    # Rank calculation
    # 2x2x6 part: 3 blocks of 2x2x2 (Strassen rank 7) -> 21
    # 2x1x6 part: 2*6 = 12
    total_rank = 3 * 7 + 12
    rank = total_rank
    
    U = [[0] * rank for _ in range(n_A)]
    V = [[0] * rank for _ in range(n_B)]
    W = [[0] * rank for _ in range(n_C)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(a_indices, b_indices, c_indices):
        nonlocal col
        # a_indices: [A00, A01, A10, A11] global indices
        # b_indices: [B00, B01, B10, B11] global indices
        # c_indices: [C00, C01, C10, C11] global indices
        
        # Strassen terms (U, V, W local)
        # M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1]
        # M2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, -1]
        # M3
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        w3 = [0, 1, 0, 1]
        # M4
        u4 = [0, 0, 0, 1]
        v4 = [0, 1, -1, 0]
        w4 = [1, 0, 1, 0]
        # M5
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0]
        # M6
        u6 = [0, 0, 1, -1]
        v6 = [1, 0, 1, 0]
        w6 = [0, 0, 0, 1]
        # M7
        u7 = [0, 1, 0, -1]
        v7 = [0, 1, 0, 1]
        w7 = [1, 0, 0, 0]
        
        terms = [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), 
                 (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]
        
        for u, v, w in terms:
            for i in range(4):
                if u[i] != 0:
                    U[a_indices[i]][col] += u[i]
                if v[i] != 0:
                    V[b_indices[i]][col] += v[i]
                if w[i] != 0:
                    W[c_indices[i]][col] += w[i]
            col += 1

    # Block 1: A(0,1) x B(0,1) -> C(0,1)
    # A indices: 0, 1, 3, 4
    # B indices: 0, 1, 6, 7
    # C indices: 0, 1, 6, 7
    add_strassen_block([0, 1, 3, 4], [0, 1, 6, 7], [0, 1, 6, 7])
    
    # Block 2: A(0,1) x B(2,3) -> C(2,3)
    # A indices: 0, 1, 3, 4
    # B indices: 2, 3, 8, 9
    # C indices: 2, 3, 8, 9
    add_strassen_block([0, 1, 3, 4], [2, 3, 8, 9], [2, 3, 8, 9])
    
    # Block 3: A(0,1) x B(4,5) -> C(4,5)
    # A indices: 0, 1, 3, 4
    # B indices: 4, 5, 10, 11
    # C indices: 4, 5, 10, 11
    add_strassen_block([0, 1, 3, 4], [4, 5, 10, 11], [4, 5, 10, 11])
    
    # Block 4: A(2) x B(2) -> C(all)
    # A indices: 2, 5 (A02, A12)
    # B indices: 12, 13, 14, 15, 16, 17 (B20..B25)
    # C indices: 0..11
    
    # A2 is 2x1, B2 is 1x6.
    # C[i,k] = A[2,i] * B[2,k]
    for i in range(2):
        a_idx = 2 + i # A02 is index 2, A12 is index 5
        for k in range(6):
            b_idx = 12 + k # B20 is index 12
            c_idx = k * n + i # W[k*n+i]
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
