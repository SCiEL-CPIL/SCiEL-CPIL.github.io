import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition manually based on the logic above.
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        # u_vec is dict or list of (index, val)
        # Convert to full vectors
        u_full = [0.0] * (n * m)
        v_full = [0.0] * (m * p)
        w_full = [0.0] * (p * n)
        
        for idx, val in u_vec:
            u_full[idx] = val
        for idx, val in v_vec:
            v_full[idx] = val
        for idx, val in w_vec:
            w_full[idx] = val
            
        U_rows.append(u_full)
        V_rows.append(v_full)
        W_rows.append(w_full)

    def add_simple(i, j, k):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])

    def add_rank1_block(i_list, j_list, k_list):
        # Adds a rank 1 term for the sum of A in i_list, j_list 
        # times sum of B in j_list, k_list
        # times sum of C in k_list, i_list
        # This is valid if the sets are rectangular and we want the full product sum
        u_vec = []
        for i in i_list:
            for j in j_list:
                u_vec.append((i * m + j, 1.0))
        
        v_vec = []
        for j in j_list:
            for k in k_list:
                v_vec.append((j * p + k, 1.0))
                
        w_vec = []
        for k in k_list:
            for i in i_list:
                w_vec.append((k * n + i, 1.0))
                
        add_term(u_vec, v_vec, w_vec)

    def add_strassen(i0, i1, j0, j1, k0, k1):
        # Adds 7 terms for 2x2x2 block
        # Local indices 0,1 map to global i0,i1 etc
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        u1 = [(i0*m+j0, 1.0), (i1*m+j1, 1.0)]
        v1 = [(j0*p+k0, 1.0), (j1*p+k1, 1.0)]
        w1 = [(k0*n+i0, 1.0), (k1*n+i1, 1.0)]
        add_term(u1, v1, w1)
        
        # M2: (A10+A11)B00 -> C10 + C11 (with -1 for C11)
        # Wait, C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes +1 to C10, -1 to C11.
        u2 = [(i1*m+j0, 1.0), (i1*m+j1, 1.0)]
        v2 = [(j0*p+k0, 1.0)]
        w2 = [(k0*n+i1, 1.0), (k1*n+i1, -1.0)]
        add_term(u2, v2, w2)
        
        # M3: A00(B10-B11) -> C01 + C11
        u3 = [(i0*m+j0, 1.0)]
        v3 = [(j1*p+k0, 1.0), (j1*p+k1, -1.0)]
        w3 = [(k0*n+i0, 1.0), (k1*n+i1, 1.0)]
        add_term(u3, v3, w3)
        
        # M4: A11(B01-B00) -> C00 + C10
        u4 = [(i1*m+j1, 1.0)]
        v4 = [(j0*p+k1, 1.0), (j0*p+k0, -1.0)]
        w4 = [(k0*n+i0, 1.0), (k0*n+i1, 1.0)]
        add_term(u4, v4, w4)
        
        # M5: (A00+A10)B11 -> C01 - C00
        # C01 = M3 + M5. C00 = M1 + M4 - M5 + M7.
        # So M5 contributes +1 to C01, -1 to C00.
        u5 = [(i0*m+j0, 1.0), (i0*m+j1, 1.0)]
        v5 = [(j1*p+k1, 1.0)]
        w5 = [(k1*n+i0, 1.0), (k0*n+i0, -1.0)]
        add_term(u5, v5, w5)
        
        # M6: (A01-A00)(B00+B10) -> C11
        # C11 = M1 - M2 + M3 + M6.
        # So M6 contributes +1 to C11.
        u6 = [(i0*m+j1, 1.0), (i0*m+j0, -1.0)]
        v6 = [(j0*p+k0, 1.0), (j1*p+k0, 1.0)]
        w6 = [(k1*n+i1, 1.0)]
        add_term(u6, v6, w6)
        
        # M7: (A10-A11)(B01+B11) -> C00
        # C00 = M1 + M4 - M5 + M7.
        # So M7 contributes +1 to C00.
        u7 = [(i1*m+j0, 1.0), (i1*m+j1, -1.0)]
        v7 = [(j0*p+k1, 1.0), (j1*p+k1, 1.0)]
        w7 = [(k0*n+i0, 1.0)]
        add_term(u7, v7, w7)

    # Construction
    # j in 0,1 (first block) and 2,3 (second block)
    for j_block in [0, 2]:
        j0 = j_block
        j1 = j_block + 1
        
        # i=0 (first row)
        # 1 x 2 x 6
        # i_list=[0], j_list=[j0, j1], k_list=[0..5]
        add_rank1_block([0], [j0, j1], [0, 1, 2, 3, 4, 5])
        
        # i=1,2 (second block of rows)
        # 2 x 2 x 6
        # i0=1, i1=2
        # Split k into 0,1,2 and 3,4,5
        for k_block in [0, 3]:
            k0 = k_block
            k1 = k_block + 1
            k2 = k_block + 2
            
            # 2 x 2 x 2 part (k0, k1)
            add_strassen(1, 2, j0, j1, k0, k1)
            
            # 2 x 2 x 1 part (k2)
            # i_list=[1,2], j_list=[j0,j1], k_list=[k2]
            add_rank1_block([1, 2], [j0, j1], [k2])

    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {"U": U, "V": V, "W": W}
