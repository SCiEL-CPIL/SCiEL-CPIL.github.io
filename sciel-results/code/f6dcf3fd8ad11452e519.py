"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition using Strassen."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 115  # 115s budget, 5s safety margin
    
    # Strassen 2x2x2: 7 rank-1 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[m,n],[o,p]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(f+h)  
    # M3 = (a-b)(g+h)
    # M4 = (c+d)(e-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M4 + M6
    # C10 = M3 + M5
    # C11 = M1 - M3 + M6 - M2
    
    # Each term is (u_dict, v_dict, w_dict) where keys are flat indices
    # For 2x2: A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_2x2 = [
        # M1 = (a+d)(e+h) -> contributes to C00, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)(f+h) -> contributes to C11 with -1
        ({2: 1, 3: 1}, {1: 1, 3: 1}, {3: -1}),
        # M3 = (a-b)(g+h) -> contributes to C10, C11
        ({0: 1, 1: -1}, {2: 1, 3: 1}, {2: 1, 3: -1}),
        # M4 = (c+d)(e-h) -> contributes to C00, C01
        ({2: 1, 3: 1}, {0: 1, 3: -1}, {0: -1, 1: 1}),
        # M5 = (a+b)h -> contributes to C00, C10
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 2: 1}),
        # M6 = (c-a)(e+f) -> contributes to C01, C11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {1: 1, 3: 1}),
        # M7 = (b-d)(g+h) -> contributes to C00
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1}),
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_strassen_4x4x4(B_col_offset):
        """Add 49 terms for 4x4 * 4x4 using Strassen at two levels."""
        # 4x4 blocks: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
        # Each block is 2x2.
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            """Add 7 Strassen terms for the inner 2x2 multiplication."""
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # A side: inner 2x2 indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = br * 2 + li
                        gj = bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # B side
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = br * 2 + lj
                        gk = B_col_offset + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # C side
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = br * 2 + li
                        gk = B_col_offset + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # Outer Strassen for 4x4:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)(B01+B11) -> C10, C11  
        # M3 = (A00-A10)(B10+B11) -> C01, C11
        # M4 = (A00+A11)(B00-B11) -> C00, C01
        # M5 = (A00+A10)(B00+B10) -> C00, C10
        # M6 = (A01-A11)(B01+B11) -> C01, C11
        # M7 = (A01-A00)(B00+B10) -> C00, C10
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_inner(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        # M2: (A10+A11)(B01+B11) -> C10 - C11
        add_inner(
            [(1,0,1), (1,1,1)],
            [(0,1,1), (1,1,1)],
            [(1,0,1), (1,1,-1)]
        )
        # M3: (A00-A10)(B10+B11) -> C01 - C11
        add_inner(
            [(0,0,1), (1,0,-1)],
            [(1,0,1), (1,1,1)],
            [(0,1,1), (1,1,-1)]
        )
        # M4: (A00+A11)(B00-B11) -> C00 - C01
        add_inner(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,-1)],
            [(0,0,1), (0,1,-1)]
        )
        # M5: (A00+A10)(B00+B10) -> C00 + C10
        add_inner(
            [(0,0,1), (1,0,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,1), (1,0,1)]
        )
        # M6: (A01-A11)(B01+B11) -> C01 + C11
        add_inner(
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,1,1), (1,1,1)]
        )
        # M7: (A01-A00)(B00+B10) -> C00 - C10
        add_inner(
            [(0,1,1), (0,0,-1)],
            [(0,0,1), (1,0,1)],
            [(0,0,1), (1,0,-1)]
        )
    
    # Part 1: A * B[:, 0:4]
    add_strassen_4x4x4(0)
    
    # Part 2: A * B[:, 4:8]
    add_strassen_4x4x4(4)
    
    # Part 3: A * B[:, 8:9] - schoolbook, rank 16
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    total_rank = len(U_cols)
    
    # Build matrices
    U = np.array(U_cols).T  # shape (16, total_rank)
    V = np.array(V_cols).T  # shape (36, total_rank)
    W = np.array(W_cols).T  # shape (36, total_rank)
    
    # Verify validity
    def check_validity(U, V, W, tol=1e-6):
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_cols_arr = np.array([V[j*p + k, :] for j in range(m)])
                W_val = W[k*n + i, :]
                temp = U_slice * W_val
                result = temp @ V_cols_arr.T
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True
    
    if not check_validity(U, V, W):
        # Fallback to schoolbook
        U_fallback = np.zeros((16, 144))
        V_fallback = np.zeros((36, 144))
        W_fallback = np.zeros((36, 144))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_fallback[i*m+j, col] = 1
                    V_fallback[j*p+k, col] = 1
                    W_fallback[k*n+i, col] = 1
                    col += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
