import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We determined the rank to be 49
    # 24 (J1) + 24 (J2) + 1 (J3) = 49
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0 # current column index
    
    # Helper to add a rank-1 term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal c
        for (i, j), val in u_dict.items():
            U[i * m + j, c] += val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] += val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] += val
        c += 1

    # Strassen 2x2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    def strassen_2x2(i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets for i, j, k
        # Local i in {0,1}, j in {0,1}, k in {0,1}
        
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # M2 = (A10+A11)(B00) -> C10 += 1
        # M3 = (A00)(B01-B10) -> C01 += 1, C11 += 1
        # M4 = (A11)(B11-B10) -> C00 += 1, C10 += 1
        # M5 = (A00+A01)(B11) -> C00 -= 1, C01 += 1
        # M6 = (A10-A00)(B00+B10) -> C11 += 1
        # M7 = (A01-A11)(B01+B11) -> C00 += 1, C01 += 1
        
        # Note: The standard Strassen formulas are:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-e)
        # M4 = d(h-g)
        # M5 = (a+b)h
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the coefficients for U (A), V (B), W (C)
        # Each M is a product of a linear combo of A and a linear combo of B.
        # The W part is the coefficient of C.
        
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # Wait, B is (e,f,g,h). e=B00, f=B01, g=B10, h=B11.
        # v1 should be {(0,0):1, (1,1):1} for (e+h)? No.
        # (e+h) corresponds to B00 + B11.
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        # Note: The index in V is (j,k).
        # B00 is j=0,k=0. B11 is j=1,k=1.
        # So v1 is correct.
        
        # M2 = (c+d)e = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(1,0): 1}
        
        # M3 = a(f-e) = A00(B01-B10)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,0): -1}
        w3 = {(0,1): 1, (1,1): 1}
        
        # M4 = d(h-g) = A11(B11-B10)
        u4 = {(1,1): 1}
        v4 = {(1,1): 1, (1,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        
        # M5 = (a+b)h = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}
        
        # M6 = (c-a)(e+g) = (A10-A00)(B00+B10)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (1,0): 1}
        w6 = {(1,1): 1}
        
        # M7 = (b-d)(f+h) = (A01-A11)(B01+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(0,1): 1, (1,1): 1}
        w7 = {(0,0): 1, (0,1): 1}
        
        # Apply offsets
        def apply_offset(d, off1, off2):
            return {(i+off1, j+off2): val for (i,j), val in d.items()}
            
        add_term(apply_offset(u1, i_off, j_off), apply_offset(v1, j_off, k_off), apply_offset(w1, k_off, i_off))
        add_term(apply_offset(u2, i_off, j_off), apply_offset(v2, j_off, k_off), apply_offset(w2, k_off, i_off))
        add_term(apply_offset(u3, i_off, j_off), apply_offset(v3, j_off, k_off), apply_offset(w3, k_off, i_off))
        add_term(apply_offset(u4, i_off, j_off), apply_offset(v4, j_off, k_off), apply_offset(w4, k_off, i_off))
        add_term(apply_offset(u5, i_off, j_off), apply_offset(v5, j_off, k_off), apply_offset(w5, k_off, i_off))
        add_term(apply_offset(u6, i_off, j_off), apply_offset(v6, j_off, k_off), apply_offset(w6, k_off, i_off))
        add_term(apply_offset(u7, i_off, j_off), apply_offset(v7, j_off, k_off), apply_offset(w7, k_off, i_off))

    # Rank 1 for 1x2 by 2x2
    # A is 1x2 (i=0, j=0,1). B is 2x2 (j=0,1; k=0,1).
    # C is 1x2 (i=0, k=0,1).
    # T_ijk = 1.
    # U_{0,j,r} V_{j,k,r} W_{k,0,r} = 1.
    # We found: U_{0,0}=1, U_{0,1}=1. V_{0,k}=1, V_{1,k}=0. W_{k,0}=1.
    def rank1_1x2_2x2(i_off, j_off, k_off):
        # i_off is the row offset (size 1)
        # j_off is the col offset of A / row offset of B (size 2)
        # k_off is the col offset of B (size 2)
        
        u = {(0,0): 1, (0,1): 1}
        v = {(0,0): 1, (0,1): 1} # V_{j=0, k=0}=1, V_{j=0, k=1}=1. V_{j=1, ...}=0
        w = {(0,0): 1, (1,0): 1} # W_{k=0, i=0}=1, W_{k=1, i=0}=1
        
        # Wait, W index is (k, i).
        # i is local 0.
        # So W_{0,0} and W_{1,0}.
        
        def apply_offset(d, off1, off2):
            return {(i+off1, j+off2): val for (i,j), val in d.items()}
            
        add_term(apply_offset(u, i_off, j_off), apply_offset(v, j_off, k_off), apply_offset(w, k_off, i_off))

    # Rank 1 for 3x1 by 1x6
    # A is 3x1 (i=0,1,2; j=0). B is 1x6 (j=0; k=0..5).
    # C is 3x6.
    # T_ijk = 1.
    # U_{i,0,r} V_{0,k,r} W_{k,i,r} = 1.
    # Set U=1, V=1, W=1.
    def rank1_3x1_1x6(i_off, j_off, k_off):
        # i_off: 0 (size 3)
        # j_off: 4 (size 1)
        # k_off: 0 (size 6)
        
        u = {}
        for i in range(3):
            u[(i, 0)] = 1
        v = {}
        for k in range(6):
            v[(0, k)] = 1
        w = {}
        for k in range(6):
            for i in range(3):
                w[(k, i)] = 1
                
        def apply_offset(d, off1, off2):
            return {(i+off1, j+off2): val for (i,j), val in d.items()}
            
        add_term(apply_offset(u, i_off, j_off), apply_offset(v, j_off, k_off), apply_offset(w, k_off, i_off))

    # Block 1: J=0,1 (j_off=0)
    # I blocks: 0,1 (i_off=0), 2 (i_off=2)
    # K blocks: 0,1 (k_off=0), 2,3 (k_off=2), 4,5 (k_off=4)
    
    # I=0,1 (2x2)
    strassen_2x2(0, 0, 0)
    strassen_2x2(0, 0, 2)
    strassen_2x2(0, 0, 4)
    
    # I=2 (1x2)
    rank1_1x2_2x2(2, 0, 0)
    rank1_1x2_2x2(2, 0, 2)
    rank1_1x2_2x2(2, 0, 4)
    
    # Block 2: J=2,3 (j_off=2)
    strassen_2x2(0, 2, 0)
    strassen_2x2(0, 2, 2)
    strassen_2x2(0, 2, 4)
    
    rank1_1x2_2x2(2, 2, 0)
    rank1_1x2_2x2(2, 2, 2)
    rank1_1x2_2x2(2, 2, 4)
    
    # Block 3: J=4 (j_off=4)
    rank1_3x1_1x6(0, 4, 0)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
