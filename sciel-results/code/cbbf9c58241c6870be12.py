"""Matrix multiplication 4x4 * 4x9 rank-minimal decomposition.

Approach: Use the 4x4x4 Strassen decomposition (rank 49) for the first two
4x4 column blocks of B, and schoolbook (rank 16) for the last 4x1 column block.
Total rank: 49 + 49 + 16 = 114.

The 4x4x4 Strassen decomposition is constructed by viewing 4x4 matrices as
2x2 blocks of 2x2 submatrices and applying Strassen at the block level, with
each block multiplication further decomposed using Strassen for 2x2x2 (rank 7).
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
    
    # Strassen 2x2x2 terms: (A_coeffs, B_coeffs, C_coeffs)
    # A entries indexed as (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B entries indexed as (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C entries indexed as (0,0)=c11, (0,1)=c12, (1,0)=c21, (1,1)=c22
    
    strassen_2x2 = [
        # s1 = (a+d)(e+h) -> c11:+1, c22:+1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s2 = (c+d)e -> c21:+1, c22:-1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # s3 = a(f-h) -> c12:+1, c22:+1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # s4 = d(g-e) -> c11:+1, c21:+1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # s5 = (a+b)h -> c11:-1, c12:+1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s6 = (c-a)(e+f) -> c22:+1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s7 = (b-d)(g+h) -> c11:+1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4."""
        
        def a_idx(br, bc, li, lj):
            """Global U index for A entry at block (br,bc), local (li,lj)."""
            gi = A_base_i + br * 2 + li
            gj = A_base_j + bc * 2 + lj
            return gi * m + gj
        
        def b_idx(br, bc, lj, lk):
            """Global V index for B entry at block (br,bc), local (lj,lk)."""
            gj = B_base_j + br * 2 + lj
            gk = B_base_k + bc * 2 + lk
            return gj * p + gk
        
        def c_idx(br, bc, li, lk):
            """Global W index for C entry at block (br,bc), local (li,lk)."""
            gi = C_base_i + br * 2 + li
            gk = C_base_k + bc * 2 + lk
            return gk * n + gi
        
        # The 7 block-level Strassen multiplications for 4x4x4:
        # Each has A_side (list of (br,bc,coeff)), B_side, and C_side (dict (br,bc)->coeff)
        
        block_mults = [
            # M1 = (A11+A22)(B11+B22) -> C11:+1, C22:+1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0):1, (1,1):1}),
            # M2 = (A21+A22)B11 -> C21:+1, C22:-1
            ([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
            # M3 = A11(B12-B22) -> C12:+1, C22:+1
            ([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1):1, (1,1):1}),
            # M4 = A22(B21-B11) -> C11:+1, C21:+1
            ([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0):1, (1,0):1}),
            # M5 = (A11+A12)B22 -> C11:-1, C12:+1
            ([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
            # M6 = (A21-A11)(B11+B12) -> C22:+1
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1):1}),
            # M7 = (A12-A22)(B21+B22) -> C11:+1
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
    
    rank = len(U_cols)
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return {"U": U, "V": V, "W": W}
