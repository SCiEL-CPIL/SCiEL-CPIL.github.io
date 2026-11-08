"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses Strassen-based block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),        # m1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # m2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # m3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # m4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # m5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # m6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # m7
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4 (rank 49)."""
        
        block_mults = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1}),
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1}),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1}),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1}),
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1}),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1}),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1}),
        ]
        
        for A_blocks, B_blocks, C_blocks in block_mults:
            for A_meta, B_meta, C_meta in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for local_idx, meta_coeff in A_meta.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for (br, bc, block_coeff) in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                
                for local_idx, meta_coeff in B_meta.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc, block_coeff) in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                
                for local_idx, meta_coeff in C_meta.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc), block_coeff in C_blocks.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 (4x4)
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (4x4)
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (4x1)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
