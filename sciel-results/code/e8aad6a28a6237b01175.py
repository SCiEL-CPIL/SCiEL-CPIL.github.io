import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    num_u = n * m  # 8
    num_v = m * p  # 28
    num_w = p * n  # 14
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Strassen for 2x2 * 2x2
    # Indices in A: i in {0,1}, j in {j0, j0+1}
    # Indices in B: j in {j0, j0+1}, k in {k0, k0+1}
    # Indices in C: i in {0,1}, k in {k0, k0+1}
    def add_strassen(a_i0, a_j0, b_j0, b_k0):
        # a00, a01, a10, a11
        # b00, b01, b10, b11
        
        # M1 = (a00+a11)(b00+b11)
        u1 = [0.0]*num_u; u1[a_i0*4+a_j0] = 1; u1[(a_i0+1)*4+(a_j0+1)] = 1
        v1 = [0.0]*num_v; v1[b_j0*7+b_k0] = 1; v1[(b_j0+1)*7+(b_k0+1)] = 1
        w1 = [0.0]*num_w; w1[b_k0*2+a_i0] = 1; w1[(b_k0+1)*2+(a_i0+1)] = -1
        add_term(u1, v1, w1)
        
        # M2 = (a10+a11)b00
        u2 = [0.0]*num_u; u2[(a_i0+1)*4+a_j0] = 1; u2[(a_i0+1)*4+(a_j0+1)] = 1
        v2 = [0.0]*num_v; v2[b_j0*7+b_k0] = 1
        w2 = [0.0]*num_w; w2[b_k0*2+(a_i0+1)] = 1; w2[(b_k0+1)*2+(a_i0+1)] = 1
        add_term(u2, v2, w2)
        
        # M3 = a00(b01-b11)
        u3 = [0.0]*num_u; u3[a_i0*4+a_j0] = 1
        v3 = [0.0]*num_v; v3[b_j0*7+(b_k0+1)] = 1; v3[(b_j0+1)*7+(b_k0+1)] = -1
        w3 = [0.0]*num_w; w3[(b_k0+1)*2+a_i0] = 1; w3[(b_k0+1)*2+(a_i0+1)] = 1
        add_term(u3, v3, w3)
        
        # M4 = a11(b10-b00)
        u4 = [0.0]*num_u; u4[(a_i0+1)*4+(a_j0+1)] = 1
        v4 = [0.0]*num_v; v4[(b_j0+1)*7+b_k0] = 1; v4[b_j0*7+b_k0] = -1
        w4 = [0.0]*num_w; w4[b_k0*2+a_i0] = 1; w4[b_k0*2+(a_i0+1)] = 1
        add_term(u4, v4, w4)
        
        # M5 = (a00+a01)b11
        u5 = [0.0]*num_u; u5[a_i0*4+a_j0] = 1; u5[a_i0*4+(a_j0+1)] = 1
        v5 = [0.0]*num_v; v5[(b_j0+1)*7+(b_k0+1)] = 1
        w5 = [0.0]*num_w; w5[b_k0*2+a_i0] = -1; w5[(b_k0+1)*2+a_i0] = 1
        add_term(u5, v5, w5)
        
        # M6 = (a10-a00)(b00+b01)
        u6 = [0.0]*num_u; u6[(a_i0+1)*4+a_j0] = 1; u6[a_i0*4+a_j0] = -1
        v6 = [0.0]*num_v; v6[b_j0*7+b_k0] = 1; v6[b_j0*7+(b_k0+1)] = 1
        w6 = [0.0]*num_w; w6[(b_k0+1)*2+(a_i0+1)] = 1
        add_term(u6, v6, w6)
        
        # M7 = (a01-a11)(b10+b11)
        u7 = [0.0]*num_u; u7[a_i0*4+(a_j0+1)] = 1; u7[(a_i0+1)*4+(a_j0+1)] = -1
        v7 = [0.0]*num_v; v7[(b_j0+1)*7+b_k0] = 1; v7[(b_j0+1)*7+(b_k0+1)] = 1
        w7 = [0.0]*num_w; w7[b_k0*2+a_i0] = 1; w7[b_k0*2+(a_i0+1)] = 1
        add_term(u7, v7, w7)

    # Schoolbook for 2x2 * 2x3
    # A (2x2), B (2x3), C (2x3)
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_02 = A_00 B_02 + A_01 B_12
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    # C_12 = A_10 B_02 + A_11 B_12
    def add_schoolbook_2x2x3(a_i0, a_j0, b_j0, b_k0):
        # Term 1: A_00 B_00 -> C_00
        u = [0.0]*num_u; u[a_i0*4+a_j0] = 1
        v = [0.0]*num_v; v[b_j0*7+b_k0] = 1
        w = [0.0]*num_w; w[b_k0*2+a_i0] = 1
        add_term(u, v, w)
        
        # Term 2: A_01 B_10 -> C_00
        u = [0.0]*num_u; u[a_i0*4+(a_j0+1)] = 1
        v = [0.0]*num_v; v[(b_j0+1)*7+b_k0] = 1
        w = [0.0]*num_w; w[b_k0*2+a_i0] = 1
        add_term(u, v, w)
        
        # Term 3: A_00 B_01 -> C_01
        u = [0.0]*num_u; u[a_i0*4+a_j0] = 1
        v = [0.0]*num_v; v[b_j0*7+(b_k0+1)] = 1
        w = [0.0]*num_w; w[(b_k0+1)*2+a_i0] = 1
        add_term(u, v, w)
        
        # Term 4: A_01 B_11 -> C_01
        u = [0.0]*num_u; u[a_i0*4+(a_j0+1)] = 1
        v = [0.0]*num_v; v[(b_j0+1)*7+(b_k0+1)] = 1
        w = [0.0]*num_w; w[(b_k0+1)*2+a_i0] = 1
        add_term(u, v, w)
        
        # Term 5: A_00 B_02 -> C_02
        u = [0.0]*num_u; u[a_i0*4+a_j0] = 1
        v = [0.0]*num_v; v[b_j0*7+(b_k0+2)] = 1
        w = [0.0]*num_w; w[(b_k0+2)*2+a_i0] = 1
        add_term(u, v, w)
        
        # Term 6: A_01 B_12 -> C_02
        u = [0.0]*num_u; u[a_i0*4+(a_j0+1)] = 1
        v = [0.0]*num_v; v[(b_j0+1)*7+(b_k0+2)] = 1
        w = [0.0]*num_w; w[(b_k0+2)*2+a_i0] = 1
        add_term(u, v, w)
        
        # Term 7: A_10 B_00 -> C_10
        u = [0.0]*num_u; u[(a_i0+1)*4+a_j0] = 1
        v = [0.0]*num_v; v[b_j0*7+b_k0] = 1
        w = [0.0]*num_w; w[b_k0*2+(a_i0+1)] = 1
        add_term(u, v, w)
        
        # Term 8: A_11 B_10 -> C_10
        u = [0.0]*num_u; u[(a_i0+1)*4+(a_j0+1)] = 1
        v = [0.0]*num_v; v[(b_j0+1)*7+b_k0] = 1
        w = [0.0]*num_w; w[b_k0*2+(a_i0+1)] = 1
        add_term(u, v, w)
        
        # Term 9: A_10 B_01 -> C_11
        u = [0.0]*num_u; u[(a_i0+1)*4+a_j0] = 1
        v = [0.0]*num_v; v[b_j0*7+(b_k0+1)] = 1
        w = [0.0]*num_w; w[(b_k0+1)*2+(a_i0+1)] = 1
        add_term(u, v, w)
        
        # Term 10: A_11 B_11 -> C_11
        u = [0.0]*num_u; u[(a_i0+1)*4+(a_j0+1)] = 1
        v = [0.0]*num_v; v[(b_j0+1)*7+(b_k0+1)] = 1
        w = [0.0]*num_w; w[(b_k0+1)*2+(a_i0+1)] = 1
        add_term(u, v, w)
        
        # Term 11: A_10 B_02 -> C_12
        u = [0.0]*num_u; u[(a_i0+1)*4+a_j0] = 1
        v = [0.0]*num_v; v[b_j0*7+(b_k0+2)] = 1
        w = [0.0]*num_w; w[(b_k0+2)*2+(a_i0+1)] = 1
        add_term(u, v, w)
        
        # Term 12: A_11 B_12 -> C_12
        u = [0.0]*num_u; u[(a_i0+1)*4+(a_j0+1)] = 1
        v = [0.0]*num_v; v[(b_j0+1)*7+(b_k0+2)] = 1
        w = [0.0]*num_w; w[(b_k0+2)*2+(a_i0+1)] = 1
        add_term(u, v, w)

    # Apply to A_left (i=0,1, j=0,1) and B_top (j=0,1, k=0..6)
    # B_top split: k=0,1 (Strassen), k=2,3 (Strassen), k=4,5,6 (Schoolbook)
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 0, 2)
    add_schoolbook_2x2x3(0, 0, 0, 4)
    
    # Apply to A_right (i=0,1, j=2,3) and B_bottom (j=2,3, k=0..6)
    # B_bottom split: k=0,1 (Strassen), k=2,3 (Strassen), k=4,5,6 (Schoolbook)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_2x2x3(0, 2, 2, 4)
    
    # Convert to arrays
    U_arr = np.array(U).T
    V_arr = np.array(V).T
    W_arr = np.array(W).T
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
