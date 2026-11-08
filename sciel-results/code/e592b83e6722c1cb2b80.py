"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Approach: Use the 4x4x4 Strassen decomposition (rank 49) for the first two
4x4 column blocks of B, and schoolbook (rank 16) for the last 4x1 column block.
Total rank: 49 + 49 + 16 = 114.
"""

import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
    
    # Strassen 2x2x2 terms
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4."""
        
        def a_idx(br, bc, li, lj):
            gi = A_base_i + br * 2 + li
            gj = A_base_j + bc * 2 + lj
            return gi * m + gj
        
        def b_idx(br, bc, lj, lk):
            gj = B_base_j + br * 2 + lj
            gk = B_base_k + bc * 2 + lk
            return gj * p + gk
        
        def c_idx(br, bc, li, lk):
            gi = C_base_i + br * 2 + li
            gk = C_base_k + bc * 2 + lk
            return gk * n + gi
        
        block_mults = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0):1, (1,1):1}),
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1):1, (1,1):1}),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0):1, (1,0):1}),
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1):1}),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0):1}),
        ]
        
        for A_side, B_side, C_side in block_mults:
            for A_coeffs, B_coeffs, C_coeffs in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                for (li, lj), mc in A_coeffs.items():
                    for (br, bc, bc_coeff) in A_side:
                        idx = a_idx(br, bc, li, lj)
                        u_vec[idx] += mc * bc_coeff
                
                for (lj, lk), mc in B_coeffs.items():
                    for (br, bc, bc_coeff) in B_side:
                        idx = b_idx(br, bc, lj, lk)
                        v_vec[idx] += mc * bc_coeff
                
                for (li, lk), mc in C_coeffs.items():
                    for (br, bc), cc_coeff in C_side.items():
                        idx = c_idx(br, bc, li, lk)
                        w_vec[idx] += mc * cc_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B[:, 0:4] (4x4 * 4x4) using Strassen, rank 49
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B[:, 4:8] (4x4 * 4x4) using Strassen, rank 49
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B[:, 8:9] (4x4 * 4x1) using schoolbook, rank 16
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return {"U": U, "V": V, "W": W}
