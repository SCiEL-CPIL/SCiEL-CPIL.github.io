"""Matrix multiplication 4,4,9 tensor decomposition - Strassen-based block decomposition."""
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Returns a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
    - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
    - Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h) -> c11(+1), c22(+1)
    # m2 = (c+d)e    -> c21(+1), c22(-1)
    # m3 = a(f-h)    -> c12(+1), c22(+1)
    # m4 = d(g-e)    -> c11(+1), c21(+1)
    # m5 = (a+b)h    -> c11(-1), c12(+1)
    # m6 = (c-a)(e+f) -> c22(+1)
    # m7 = (b-d)(g+h) -> c11(+1)
    
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(B_col_offset):
        """
        Add Strassen terms for a 4x4*4x4 block.
        B is at columns B_col_offset .. B_col_offset+3.
        """
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = br * 2 + li
                        gj = bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = br * 2 + lj
                        gk = B_col_offset + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = br * 2 + li
                        gk = B_col_offset + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # Strassen 4x4 block multiplications
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
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
        
    # Apply Strassen for first 4x4 block of B (cols 0-3)
    add_strassen_4x4(0)
    
    # Apply Strassen for second 4x4 block of B (cols 4-7)
    add_strassen_4x4(4)
    
    # Schoolbook for last 1x4 block of B (col 8)
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
    
    # Convert to required format
    U_out = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V_out = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W_out = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
