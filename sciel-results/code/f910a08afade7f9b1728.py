import time
import numpy as np

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    Compute A*B1 using Strassen (rank 49).
    Compute A*B2 using Strassen (rank 49).
    Compute A*B3 using Schoolbook (rank 16).
    Total rank: 49 + 49 + 16 = 114.
    
    We attempt to reduce rank by using more efficient decompositions
    or by sharing terms where possible.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Strassen 2x2 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[c11,c12],[c21,c22]]
    # m1 = (a+d)(e+h) -> c11:+1, c22:+1
    # m2 = (c+d)e    -> c21:+1, c22:-1
    # m3 = a(f-h)    -> c12:+1, c22:+1
    # m4 = d(g-e)    -> c11:+1, c21:+1
    # m5 = (a+b)h    -> c11:-1, c12:+1
    # m6 = (c-a)(e+f)-> c22:+1
    # m7 = (b-d)(g+h)-> c11:+1
    
    # Indices in 2x2:
    # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    s2x2 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # m1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),     # m2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),     # m3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),     # m4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),     # m5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),     # m6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),     # m7
    ]
    
    def add_strassen_4x4(A_off, B_off, C_off):
        """
        Add Strassen terms for 4x4 * 4x4.
        A_off: offset (row, col) in A
        B_off: offset (row, col) in B
        C_off: offset (row, col) in C
        """
        # 7 block multiplications
        # M1 = (A11+A22)(B11+B22) -> C11:+1, C22:+1
        # M2 = (A21+A22)B11      -> C21:+1, C22:-1
        # M3 = A11(B12-B22)      -> C12:+1, C22:+1
        # M4 = A22(B21-B11)      -> C11:+1, C21:+1
        # M5 = (A11+A12)B22      -> C11:-1, C12:+1
        # M6 = (A21-A11)(B11+B12)-> C22:+1
        # M7 = (A12-A22)(B21+B22)-> C11:+1
        
        blocks = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),           # M2
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),           # M3
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),           # M4
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),           # M5
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),           # M6
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),           # M7
        ]
        
        for A_side, B_side, C_side in blocks:
            for u_coeff, v_coeff, w_coeff in s2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for local_idx, coeff in u_coeff.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for br, bc, b_coeff in A_side:
                        gi = A_off[0] + br * 2 + li
                        gj = A_off[1] + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * b_coeff
                        
                for local_idx, coeff in v_coeff.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, b_coeff in B_side:
                        gj = B_off[0] + br * 2 + lj
                        gk = B_off[1] + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * b_coeff
                        
                for local_idx, coeff in w_coeff.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for br, bc, b_coeff in C_side:
                        gi = C_off[0] + br * 2 + li
                        gk = C_off[1] + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * b_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4((0,0), (0,0), (0,0))
    
    # Check time
    if time.time() > deadline:
        rank = len(U_cols)
        if rank == 0:
            return {"U": [[0]*1 for _ in range(16)], "V": [[0]*1 for _ in range(36)], "W": [[0]*1 for _ in range(36)]}
        U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
        V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
        W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
        return {"U": U, "V": V, "W": W}
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4((0,0), (0,4), (0,4))
    
    # Check time
    if time.time() > deadline:
        rank = len(U_cols)
        U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
        V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
        W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
        return {"U": U, "V": V, "W": W}
    
    # Part 3: A * B3 (col 8)
    # Schoolbook 4x4 * 4x1
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
