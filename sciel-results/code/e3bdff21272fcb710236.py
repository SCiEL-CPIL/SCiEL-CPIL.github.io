import time
import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    - Partition B into three column blocks: B1 (4x4), B2 (4x4), B3 (4x1).
    - Use Strassen (rank 49) for A*B1 and A*B2.
    - Use schoolbook (rank 16) for A*B3.
    - Total: 49 + 49 + 16 = 114.
    """
    
    n, m, p = 4, 4, 9
    deadline = time.time() + 120
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2x2 terms: each is (u_sparsity, v_sparsity, w_sparsity)
    # 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def build_4x4_strassen(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        """
        Build 49 terms for 4x4 * 4x4 Strassen decomposition.
        Uses 7 outer Strassen products, each being a 2x2 * 2x2 Strassen (7 terms).
        """
        # Outer Strassen M1..M7 with block assignments
        # Each M_i: (A_blocks, B_blocks, C_blocks) where each block list has (row, col, coeff)
        outer_m = [
            # M1 = (A11+A22)(B11+B22) -> C11 + C22
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A21+A22)B11 -> C21 - C22
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3 = A11(B12-B22) -> C12 + C22
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4 = A22(B21-B11) -> C11 + C21
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5 = (A11+A12)B22 -> -C11 + C12
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A21-A11)(B11+B12) -> C22
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7 = (A12-A22)(B21+B22) -> C11
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for A_blocks, B_blocks, C_blocks in outer_m:
            for u_sp, v_sp, w_sp in strassen_2x2:
                u_vec = [0.0] * 16
                v_vec = [0.0] * 36
                w_vec = [0.0] * 36
                
                for loc, cf in u_sp.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bcf in A_blocks:
                        gi = a_base_i + br * 2 + li
                        gj = a_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += cf * bcf
                
                for loc, cf in v_sp.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bcf in B_blocks:
                        gj = b_base_j + br * 2 + lj
                        gk = b_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += cf * bcf
                
                for loc, cf in w_sp.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bcf in C_blocks:
                        gi = c_base_i + br * 2 + li
                        gk = c_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += cf * bcf
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 (cols 0-3)
    build_4x4_strassen(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    build_4x4_strassen(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8) - schoolbook 4x4 * 4x1
    for i in range(n):
        for j in range(m):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 36
            w_vec = [0.0] * 36
            u_vec[i * m + j] = 1.0
            v_vec[j * p + 8] = 1.0
            w_vec[8 * n + i] = 1.0
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
