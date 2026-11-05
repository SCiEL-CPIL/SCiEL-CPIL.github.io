import time
import numpy as np

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen for the 4x4*4x4 parts and schoolbook for 4x4*4x1.
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    deadline = time.time() + 140  # safety margin
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2 rank-1 terms: (u_support, v_support, w_support)
    # Each support is a dict mapping local 2x2 index to coefficient
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_4x4(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        """Add 49 terms for 4x4 * 4x4 using Strassen."""
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_sup, v_sup, w_sup in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                for (li, lj), coeff in u_sup.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = a_i0 + br * 2 + li
                        gj = a_j0 + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                for (lj, lk), coeff in v_sup.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = b_j0 + br * 2 + lj
                        gk = b_k0 + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                for (li, lk), coeff in w_sup.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = c_i0 + br * 2 + li
                        gk = c_k0 + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        # M1 = (A11+A22)(B11+B22)
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        # M2 = (A21+A22)B11
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        # M3 = A11(B12-B22)
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        # M4 = A22(B21-B11)
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        # M5 = (A11+A12)B22
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        # M6 = (A21-A11)(B11+B12)
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        # M7 = (A12-A22)(B21+B22)
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
    # Part 1: A * B[:, 0:4]
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B[:, 4:8]
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B[:, 8], schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    
    # Transpose to get row-major matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
