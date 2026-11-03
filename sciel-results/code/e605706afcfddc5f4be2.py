import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a rank-1 term for a 2x2x2 block
    # a_idx, b_idx, c_idx, d_idx: indices in U for A elements
    # e_idx, f_idx, g_idx, h_idx: indices in V for B elements
    # c00_idx, c01_idx, c10_idx, c11_idx: indices in W for C elements
    # term_idx: column index in U, V, W
    def add_strassen_term(term_idx, a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, c00_idx, c01_idx, c10_idx, c11_idx):
        # M1 = (a+d)(e+h)
        U[a_idx, term_idx] += 1
        U[d_idx, term_idx] += 1
        V[e_idx, term_idx] += 1
        V[h_idx, term_idx] += 1
        W[c00_idx, term_idx] += 1
        W[c11_idx, term_idx] += 1
        
        # M2 = (c+d)e
        term_idx += 1
        U[c_idx, term_idx] += 1
        U[d_idx, term_idx] += 1
        V[e_idx, term_idx] += 1
        W[c10_idx, term_idx] += 1
        
        # M3 = a(f-h)
        term_idx += 1
        U[a_idx, term_idx] += 1
        V[f_idx, term_idx] += 1
        V[h_idx, term_idx] -= 1
        W[c01_idx, term_idx] += 1
        W[c11_idx, term_idx] += 1
        
        # M4 = d(g-e)
        term_idx += 1
        U[d_idx, term_idx] += 1
        V[g_idx, term_idx] += 1
        V[e_idx, term_idx] -= 1
        W[c00_idx, term_idx] += 1
        W[c10_idx, term_idx] += 1
        
        # M5 = (a+b)h
        term_idx += 1
        U[a_idx, term_idx] += 1
        U[b_idx, term_idx] += 1
        V[h_idx, term_idx] += 1
        W[c00_idx, term_idx] -= 1
        W[c01_idx, term_idx] += 1
        
        # M6 = (c-a)(e+f)
        term_idx += 1
        U[c_idx, term_idx] += 1
        U[a_idx, term_idx] -= 1
        V[e_idx, term_idx] += 1
        V[f_idx, term_idx] += 1
        W[c11_idx, term_idx] += 1
        
        # M7 = (b-d)(g+h)
        term_idx += 1
        U[b_idx, term_idx] += 1
        U[d_idx, term_idx] -= 1
        V[g_idx, term_idx] += 1
        V[h_idx, term_idx] += 1
        W[c00_idx, term_idx] += 1
        
        return term_idx + 1

    # Helper to add rank-1 terms for a 2x2x1 block (schoolbook)
    # C0 = a*e + b*g
    # C1 = c*e + d*h
    def add_2x2x1_term(term_idx, a_idx, b_idx, c_idx, d_idx, e_idx, g_idx, c0_idx, c1_idx):
        # Term 1: a*e
        U[a_idx, term_idx] += 1
        V[e_idx, term_idx] += 1
        W[c0_idx, term_idx] += 1
        
        # Term 2: b*g
        term_idx += 1
        U[b_idx, term_idx] += 1
        V[g_idx, term_idx] += 1
        W[c0_idx, term_idx] += 1
        
        # Term 3: c*e
        term_idx += 1
        U[c_idx, term_idx] += 1
        V[e_idx, term_idx] += 1
        W[c1_idx, term_idx] += 1
        
        # Term 4: d*h -- Wait, d is A[1,1], h is B[1,0]?
        # A = [[a, b], [c, d]]
        # B = [[e, ?], [g, ?]] -- Here B is 2x1, so B = [[e], [g]]
        # C = A B = [[a*e + b*g], [c*e + d*g]]
        # My previous mapping:
        # e_idx: B[0,0]
        # g_idx: B[1,0]
        # So term 4 is d*g
        U[d_idx, term_idx] += 1
        V[g_idx, term_idx] += 1
        W[c1_idx, term_idx] += 1
        
        return term_idx + 1

    current_term = 0
    
    # A indices:
    # A_L: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # A_R: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    
    a_L = [0, 1, 2, 3] # a, b, c, d
    a_R = [4, 5, 6, 7] # a, b, c, d
    
    # B indices:
    # B is 4x7. Flat index j*p + k.
    # Block 1 (cols 0,1):
    # B_1L: B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # B_1R: B[2,0]=14, B[2,1]=15, B[3,0]=21, B[3,1]=22
    
    # C indices:
    # W is p*n = 14. Flat index k*n + i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    # C[0,4]=8, C[0,5]=9, C[1,4]=10, C[1,5]=11
    # C[0,6]=12, C[1,6]=13

    # Process Block 1 (Cols 0,1)
    # A_L * B_1L
    b_1L_e = 0   # B[0,0]
    b_1L_f = 1   # B[0,1]
    b_1L_g = 7   # B[1,0]
    b_1L_h = 8   # B[1,1]
    c_1L_00 = 0  # C[0,0]
    c_1L_01 = 1  # C[0,1]
    c_1L_10 = 2  # C[1,0]
    c_1L_11 = 3  # C[1,1]
    current_term = add_strassen_term(current_term, *a_L, b_1L_e, b_1L_f, b_1L_g, b_1L_h, c_1L_00, c_1L_01, c_1L_10, c_1L_11)
    
    # A_R * B_1R
    b_1R_e = 14  # B[2,0]
    b_1R_f = 15  # B[2,1]
    b_1R_g = 21  # B[3,0]
    b_1R_h = 22  # B[3,1]
    current_term = add_strassen_term(current_term, *a_R, b_1R_e, b_1R_f, b_1R_g, b_1R_h, c_1L_00, c_1L_01, c_1L_10, c_1L_11)

    # Process Block 2 (Cols 2,3)
    # A_L * B_2L
    b_2L_e = 2   # B[0,2]
    b_2L_f = 3   # B[0,3]
    b_2L_g = 9   # B[1,2]
    b_2L_h = 10  # B[1,3]
    c_2L_00 = 4  # C[0,2]
    c_2L_01 = 5  # C[0,3]
    c_2L_10 = 6  # C[1,2]
    c_2L_11 = 7  # C[1,3]
    current_term = add_strassen_term(current_term, *a_L, b_2L_e, b_2L_f, b_2L_g, b_2L_h, c_2L_00, c_2L_01, c_2L_10, c_2L_11)
    
    # A_R * B_2R
    b_2R_e = 16  # B[2,2]
    b_2R_f = 17  # B[2,3]
    b_2R_g = 23  # B[3,2]
    b_2R_h = 24  # B[3,3]
    current_term = add_strassen_term(current_term, *a_R, b_2R_e, b_2R_f, b_2R_g, b_2R_h, c_2L_00, c_2L_01, c_2L_10, c_2L_11)

    # Process Block 3 (Cols 4,5)
    # A_L * B_3L
    b_3L_e = 4   # B[0,4]
    b_3L_f = 5   # B[0,5]
    b_3L_g = 11  # B[1,4]
    b_3L_h = 12  # B[1,5]
    c_3L_00 = 8  # C[0,4]
    c_3L_01 = 9  # C[0,5]
    c_3L_10 = 10 # C[1,4]
    c_3L_11 = 11 # C[1,5]
    current_term = add_strassen_term(current_term, *a_L, b_3L_e, b_3L_f, b_3L_g, b_3L_h, c_3L_00, c_3L_01, c_3L_10, c_3L_11)
    
    # A_R * B_3R
    b_3R_e = 18  # B[2,4]
    b_3R_f = 19  # B[2,5]
    b_3R_g = 25  # B[3,4]
    b_3R_h = 26  # B[3,5]
    current_term = add_strassen_term(current_term, *a_R, b_3R_e, b_3R_f, b_3R_g, b_3R_h, c_3L_00, c_3L_01, c_3L_10, c_3L_11)

    # Process Block 4 (Col 6)
    # A_L * B_4L (2x2x1)
    b_4L_e = 6   # B[0,6]
    b_4L_g = 13  # B[1,6]
    c_4L_0 = 12  # C[0,6]
    c_4L_1 = 13  # C[1,6]
    current_term = add_2x2x1_term(current_term, *a_L[:2], a_L[2], a_L[3], b_4L_e, b_4L_g, c_4L_0, c_4L_1)
    # Note: a_L is [a,b,c,d]. a_L[:2] is a,b. a_L[2] is c. a_L[3] is d.
    # My add_2x2x1_term signature: (term_idx, a_idx, b_idx, c_idx, d_idx, e_idx, g_idx, c0_idx, c1_idx)
    # So I passed a, b, c, d correctly.
    
    # A_R * B_4R (2x2x1)
    b_4R_e = 20  # B[2,6]
    b_4R_g = 27  # B[3,6]
    current_term = add_2x2x1_term(current_term, *a_R[:2], a_R[2], a_R[3], b_4R_e, b_4R_g, c_4L_0, c_4L_1)

    return {'U': U, 'V': V, 'W': W}
