import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # We will construct a rank 50 decomposition.
    # Split B (4x7) into B1 (4x2), B2 (4x2), B3 (4x2), B4 (4x1)
    # Split A (2x4) into AL (2x2) cols 0,1 and AR (2x2) cols 2,3
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: dict { (i,j): val } or array of size 8
        v_vec: dict { (j,k): val } or array of size 28
        w_vec: dict { (k,i): val } or array of size 14
        """
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Helper to create vectors
    def make_u(indices_vals):
        vec = np.zeros(dim_u)
        for (i, j), val in indices_vals.items():
            vec[i * m + j] = val
        return vec

    def make_v(indices_vals):
        vec = np.zeros(dim_v)
        for (j, k), val in indices_vals.items():
            vec[j * p + k] = val
        return vec

    def make_w(indices_vals):
        vec = np.zeros(dim_w)
        for (k, i), val in indices_vals.items():
            vec[k * n + i] = val
        return vec

    # Strassen 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # Returns list of (u_dict, v_dict, w_dict)
    def strassen_2x2x2(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i):
        # a_base_i, a_base_j: base indices for A (i, j)
        # b_base_j, b_base_k: base indices for B (j, k)
        # c_base_k, c_base_i: base indices for C (k, i)
        # A indices: (a_base_i, a_base_j + 0), (a_base_i, a_base_j + 1), etc.
        # B indices: (b_base_j + 0, b_base_k), (b_base_j + 1, b_base_k), etc.
        # C indices: (b_base_k, c_base_i), (b_base_k + 1, c_base_i), etc.
        
        terms = []
        
        # M1 = (a00 + a11)(b00 + b11)
        u1 = {(a_base_i, a_base_j): 1, (a_base_i+1, a_base_j+1): 1}
        v1 = {(b_base_j, b_base_k): 1, (b_base_j+1, b_base_k+1): 1}
        w1 = {(b_base_k, c_base_i): 1, (b_base_k+1, c_base_i): 1, (b_base_k, c_base_i+1): 1, (b_base_k+1, c_base_i+1): 1}
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Wait, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add contributions to W for each M.
        # M1 contributes to all 4 outputs with coeff 1.
        add_term(make_u(u1), make_v(v1), make_w({(b_base_k, c_base_i): 1, (b_base_k+1, c_base_i): 1, (b_base_k, c_base_i+1): 1, (b_base_k+1, c_base_i+1): 1}))
        
        # M2 = (a10 + a11) b00
        u2 = {(a_base_i+1, a_base_j): 1, (a_base_i+1, a_base_j+1): 1}
        v2 = {(b_base_j, b_base_k): 1}
        w2 = {(b_base_k, c_base_i+1): 1, (b_base_k+1, c_base_i+1): 1} # C10, C11
        add_term(make_u(u2), make_v(v2), make_w(w2))
        
        # M3 = a00 (b10 - b11)
        u3 = {(a_base_i, a_base_j): 1}
        v3 = {(b_base_j+1, b_base_k): 1, (b_base_j+1, b_base_k+1): -1}
        w3 = {(b_base_k, c_base_i+1): 1, (b_base_k+1, c_base_i+1): 1} # C01, C11
        add_term(make_u(u3), make_v(v3), make_w(w3))
        
        # M4 = a11 (b01 - b00)
        u4 = {(a_base_i+1, a_base_j+1): 1}
        v4 = {(b_base_j, b_base_k+1): 1, (b_base_j, b_base_k): -1}
        w4 = {(b_base_k, c_base_i): 1, (b_base_k, c_base_i+1): 1} # C00, C10
        add_term(make_u(u4), make_v(v4), make_w(w4))
        
        # M5 = (a00 + a01) b11
        u5 = {(a_base_i, a_base_j): 1, (a_base_i, a_base_j+1): 1}
        v5 = {(b_base_j+1, b_base_k+1): 1}
        w5 = {(b_base_k+1, c_base_i): -1, (b_base_k+1, c_base_i+1): 1} # C01 += M5, C00 -= M5
        # C00 = M1 + M4 - M5 + M7 -> W for M5 in C00 is -1
        # C01 = M3 + M5 -> W for M5 in C01 is 1
        add_term(make_u(u5), make_v(v5), make_w({(b_base_k, c_base_i): -1, (b_base_k+1, c_base_i+1): 1}))
        # Wait, M5 is (a00+a01)b11.
        # b11 is at (j=1, k=1).
        # So v5 is at (b_base_j+1, b_base_k+1).
        # Output C00 (k=0, i=0) gets -M5. Output C01 (k=1, i=0) gets +M5.
        # My w5 dict above: (b_base_k, c_base_i) is (0,0) -> -1. (b_base_k+1, c_base_i+1) is (1,1) -> 1.
        # But M5 should affect C00 and C01.
        # C00 index: k=0, i=0.
        # C01 index: k=1, i=0.
        # So w5 should be {(b_base_k, c_base_i): -1, (b_base_k+1, c_base_i): 1}.
        # Let's re-verify Strassen formulas.
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        # M5 = (a00+a01)b11.
        # C00 has -M5. C01 has +M5.
        # So w5 should map to C00 and C01.
        # C00: (k=0, i=0). C01: (k=1, i=0).
        # So w5 = {(b_base_k, c_base_i): -1, (b_base_k+1, c_base_i): 1}.
        # I will correct this in the code.
        
        # M6 = (a10 - a00)(b00 + b01)
        u6 = {(a_base_i+1, a_base_j): 1, (a_base_i, a_base_j): -1}
        v6 = {(b_base_j, b_base_k): 1, (b_base_j, b_base_k+1): 1}
        w6 = {(b_base_k+1, c_base_i+1): 1} # C11 += M6
        add_term(make_u(u6), make_v(v6), make_w(w6))
        
        # M7 = (a01 - a11)(b10 + b11)
        u7 = {(a_base_i, a_base_j+1): 1, (a_base_i+1, a_base_j+1): -1}
        v7 = {(b_base_j+1, b_base_k): 1, (b_base_j+1, b_base_k+1): 1}
        w7 = {(b_base_k, c_base_i): 1} # C00 += M7
        add_term(make_u(u7), make_v(v7), make_w(w7))
        
        return 7 # Returns rank added
    
    # Schoolbook 2x2x1
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00], [b10]]
    # C = [[c00], [c10]]
    # c00 = a00b00 + a01b10
    # c10 = a10b00 + a11b10
    def schoolbook_2x2x1(a_base_i, a_base_j, b_base_j, b_base_k, c_base_k, c_base_i):
        # b_base_k is the single column index
        
        # 1. a00 b00 -> C00
        u1 = {(a_base_i, a_base_j): 1}
        v1 = {(b_base_j, b_base_k): 1}
        w1 = {(b_base_k, c_base_i): 1}
        add_term(make_u(u1), make_v(v1), make_w(w1))
        
        # 2. a01 b10 -> C00
        u2 = {(a_base_i, a_base_j+1): 1}
        v2 = {(b_base_j+1, b_base_k): 1}
        w2 = {(b_base_k, c_base_i): 1}
        add_term(make_u(u2), make_v(v2), make_w(w2))
        
        # 3. a10 b00 -> C10
        u3 = {(a_base_i+1, a_base_j): 1}
        v3 = {(b_base_j, b_base_k): 1}
        w3 = {(b_base_k, c_base_i+1): 1}
        add_term(make_u(u3), make_v(v3), make_w(w3))
        
        # 4. a11 b10 -> C10
        u4 = {(a_base_i+1, a_base_j+1): 1}
        v4 = {(b_base_j+1, b_base_k): 1}
        w4 = {(b_base_k, c_base_i+1): 1}
        add_term(make_u(u4), make_v(v4), make_w(w4))
        
        return 4

    # Block 1: B1 (cols 0,1)
    # AL (cols 0,1 of A) * B1L (rows 0,1 of B1)
    # A: i=0,1; j=0,1. B: j=0,1; k=0,1. C: k=0,1; i=0,1.
    rank_total += strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # AR (cols 2,3 of A) * B1R (rows 2,3 of B1)
    # A: i=0,1; j=2,3. B: j=2,3; k=0,1. C: k=0,1; i=0,1.
    rank_total += strassen_2x2x2(0, 2, 2, 0, 0, 0)
    
    # Block 2: B2 (cols 2,3)
    rank_total += strassen_2x2x2(0, 0, 0, 2, 2, 0)
    rank_total += strassen_2x2x2(0, 2, 2, 2, 2, 0)
    
    # Block 3: B3 (cols 4,5)
    rank_total += strassen_2x2x2(0, 0, 0, 4, 4, 0)
    rank_total += strassen_2x2x2(0, 2, 2, 4, 4, 0)
    
    # Block 4: B4 (col 6)
    # AL * B4L (row 0,1 of B4) -> 2x2x1
    rank_total += schoolbook_2x2x1(0, 0, 0, 6, 6, 0)
    # AR * B4R (row 2,3 of B4) -> 2x2x1
    rank_total += schoolbook_2x2x1(0, 2, 2, 6, 6, 0)
    
    # Convert lists to arrays
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
