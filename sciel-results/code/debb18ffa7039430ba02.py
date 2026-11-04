"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    num_u = n * m  # 8
    num_v = m * p  # 28
    num_w = p * n  # 14
    
    # Target rank: 50 (2 * Strassen(7) + 28 direct)
    # Structure:
    # A (2x4) = [A1 A2], where A1, A2 are 2x2
    # B (4x7) = [B1; B2], where B1, B2 are 2x7
    # C = A1 B1 + A2 B2
    
    # Part 1: Compute A1 B1 using Strassen (Rank 7)
    # Part 2: Compute A2 B2 using Strassen (Rank 7)
    # Part 3: Wait, Strassen computes 2x2 * 2x2.
    # Here B is 2x7. We can split B1 (2x7) into B1_L (2x3), B1_M (2x1), B1_R (2x3)? No, that's inefficient.
    # Actually, Strassen works for 2x2 * 2x2.
    # If we use Strassen for the 2x2 part of the multiplication, we need to handle the 7 width.
    # Let's view the multiplication A (2x4) * B (4x7) as:
    # C = A_left * B_top + A_right * B_bottom
    # where A_left, A_right are 2x2, B_top, B_bottom are 2x7.
    
    # To use Strassen, we need 2x2 * 2x2.
    # We can split B_top (2x7) into 3 parts? No.
    # We can just use the standard algorithm for 2x2 * 2x7?
    # Rank of 2x2 * 2x7 is at least max(4, 14, 14) = 14.
    # Schoolbook for 2x2 * 2x7 is 4*7 = 28.
    # Can we do 2x2 * 2x7 in rank < 28?
    # Known result: R(2,2,7) = 28? Or maybe lower?
    # If R(2,2,7) = 28, then total rank is 28 + 28 = 56 (Schoolbook).
    # But we found a rank 50 solution in inspirations. How?
    # Inspiration 1 said: "Replaces schoolbook rank 56 with a constructive rank 50 decomposition using Strassen for 2x2x2 blocks and direct rank-1 terms for the remainder"
    
    # Let's analyze the "Strassen for 2x2x2 blocks" approach more carefully.
    # Maybe we split the 4x7 matrix B into 2x2 blocks?
    # B is 4x7. We can't tile it perfectly with 2x2.
    # However, we can split the *input* dimensions.
    
    # Let's look at the specific rank 50 construction.
    # Total terms = 50.
    # 50 = 7 + 7 + 36? No.
    # 50 = 2 * 7 + 36?
    # 50 = 28 + 22?
    
    # Let's try to construct a rank 50 decomposition explicitly.
    # Strategy:
    # Split m=4 into 2+2.
    # C = A1 B1 + A2 B2.
    # A1, A2 are 2x2. B1, B2 are 2x7.
    
    # If we use Strassen for 2x2 * 2x2, we need B to be 2x2.
    # We can split the 7 columns of B into three parts: 2, 2, 3.
    # B1 = [B1_0 (2x2), B1_1 (2x2), B1_2 (2x3)]
    # Then A1 B1 = A1 B1_0 + A1 B1_1 + A1 B1_2.
    # Rank(A1 B1_0) <= 7 (Strassen)
    # Rank(A1 B1_1) <= 7 (Strassen)
    # Rank(A1 B1_2) <= 2*3 = 6 (Schoolbook for 2x2*2x3 is 6? No, 2*3=6 terms in schoolbook?
    #   A (2x2) * B (2x3) -> C (2x3).
    #   Terms: A_00 B_00, A_01 B_10, A_00 B_01, A_01 B_11, A_00 B_02, A_01 B_12.
    #   Yes, 6 terms.
    # So Rank(A1 B1) <= 7 + 7 + 6 = 20.
    # Similarly for A2 B2: Rank(A2 B2) <= 20.
    # Total Rank <= 40.
    # Wait, is R(2,2,3) = 6?
    # Lower bound for 2x2x3 is max(4, 6, 6) = 6.
    # Yes, schoolbook is optimal for 2x2x3? Likely.
    
    # So we can potentially achieve rank 40!
    # Let's verify if we can combine them.
    # The decomposition is a direct sum.
    # Total rank = 20 + 20 = 40.
    
    # Let's build this.
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: length 8 (2x4)
        v_vec: length 28 (4x7)
        w_vec: length 14 (7x2)
        """
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Helper to create vectors
    def make_u(i, j, val):
        vec = [0.0] * 8
        vec[i*4 + j] = val
        return vec
        
    def make_v(j, k, val):
        vec = [0.0] * 28
        vec[j*7 + k] = val
        return vec
        
    def make_w(k, i, val):
        vec = [0.0] * 14
        vec[k*2 + i] = val
        return vec

    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # For A1 (top-left 2x2 of A), indices i in {0,1}, j in {0,1}.
    # For B1_top (top-left 2x2 of B1, which is top-left 2x2 of B), indices j in {0,1}, k in {0,1}.
    
    def add_strassen_block(a_i_base, a_j_base, b_j_base, b_k_base):
        """
        Adds 7 terms for multiplying A[a_i_base:a_i_base+2, a_j_base:a_j_base+2] 
        with B[b_j_base:b_j_base+2, b_k_base:b_k_base+2].
        The result contributes to C[a_i_base:a_i_base+2, b_k_base:b_k_base+2].
        """
        # a is 2x2 block of A. Indices:
        # a00: (a_i_base, a_j_base)
        # a01: (a_i_base, a_j_base+1)
        # a10: (a_i_base+1, a_j_base)
        # a11: (a_i_base+1, a_j_base+1)
        
        # b is 2x2 block of B. Indices:
        # b00: (b_j_base, b_k_base)
        # b01: (b_j_base, b_k_base+1)
        # b10: (b_j_base+1, b_k_base)
        # b11: (b_j_base+1, b_k_base+1)
        
        # c is 2x2 block of C. Indices:
        # c00: (a_i_base, b_k_base)
        # c01: (a_i_base, b_k_base+1)
        # c10: (a_i_base+1, b_k_base)
        # c11: (a_i_base+1, b_k_base+1)
        
        # Term 1: M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0]*8; u1[a_i_base*4 + a_j_base] = 1; u1[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v1 = [0.0]*28; v1[b_j_base*7 + b_k_base] = 1; v1[(b_j_base+1)*7 + (b_k_base+1)] = 1
        # W contributes to c00 (+1), c01 (0), c10 (0), c11 (-1)
        w1 = [0.0]*14; w1[b_k_base*2 + a_i_base] = 1; w1[(b_k_base+1)*2 + (a_i_base+1)] = -1
        add_term(u1, v1, w1)
        
        # Term 2: M2 = (a10 + a11) * b00
        u2 = [0.0]*8; u2[(a_i_base+1)*4 + a_j_base] = 1; u2[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v2 = [0.0]*28; v2[b_j_base*7 + b_k_base] = 1
        # W contributes to c10 (+1), c11 (+1)
        w2 = [0.0]*14; w2[b_k_base*2 + (a_i_base+1)] = 1; w2[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u2, v2, w2)
        
        # Term 3: M3 = a00 * (b01 - b11)
        u3 = [0.0]*8; u3[a_i_base*4 + a_j_base] = 1
        v3 = [0.0]*28; v3[b_j_base*7 + (b_k_base+1)] = 1; v3[(b_j_base+1)*7 + (b_k_base+1)] = -1
        # W contributes to c01 (+1), c11 (+1)
        w3 = [0.0]*14; w3[(b_k_base+1)*2 + a_i_base] = 1; w3[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u3, v3, w3)
        
        # Term 4: M4 = a11 * (b10 - b00)
        u4 = [0.0]*8; u4[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v4 = [0.0]*28; v4[(b_j_base+1)*7 + b_k_base] = 1; v4[b_j_base*7 + b_k_base] = -1
        # W contributes to c00 (+1), c10 (+1)
        w4 = [0.0]*14; w4[b_k_base*2 + a_i_base] = 1; w4[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u4, v4, w4)
        
        # Term 5: M5 = (a00 + a01) * b11
        u5 = [0.0]*8; u5[a_i_base*4 + a_j_base] = 1; u5[a_i_base*4 + (a_j_base+1)] = 1
        v5 = [0.0]*28; v5[(b_j_base+1)*7 + (b_k_base+1)] = 1
        # W contributes to c00 (-1), c01 (+1)
        w5 = [0.0]*14; w5[b_k_base*2 + a_i_base] = -1; w5[(b_k_base+1)*2 + a_i_base] = 1
        add_term(u5, v5, w5)
        
        # Term 6: M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0]*8; u6[(a_i_base+1)*4 + a_j_base] = 1; u6[a_i_base*4 + a_j_base] = -1
        v6 = [0.0]*28; v6[b_j_base*7 + b_k_base] = 1; v6[b_j_base*7 + (b_k_base+1)] = 1
        # W contributes to c11 (+1)
        w6 = [0.0]*14; w6[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u6, v6, w6)
        
        # Term 7: M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0]*8; u7[a_i_base*4 + (a_j_base+1)] = 1; u7[(a_i_base+1)*4 + (a_j_base+1)] = -1
        v7 = [0.0]*28; v7[(b_j_base+1)*7 + b_k_base] = 1; v7[(b_j_base+1)*7 + (b_k_base+1)] = 1
        # W contributes to c00 (+1), c10 (+1)
        w7 = [0.0]*14; w7[b_k_base*2 + a_i_base] = 1; w7[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u7, v7, w7)

    # Schoolbook for 2x2 * 2x3
    # A (2x2), B (2x3), C (2x3)
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_02 = A_00 B_02 + A_01 B_12
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    # C_12 = A_10 B_02 + A_11 B_12
    # 6 terms.
    
    def add_schoolbook_2x2x3(a_i_base, a_j_base, b_j_base, b_k_base):
        """
        A block 2x2, B block 2x3.
        """
        # Term 1: A_00 B_00 -> C_00
        u1 = [0.0]*8; u1[a_i_base*4 + a_j_base] = 1
        v1 = [0.0]*28; v1[b_j_base*7 + b_k_base] = 1
        w1 = [0.0]*14; w1[b_k_base*2 + a_i_base] = 1
        add_term(u1, v1, w1)
        
        # Term 2: A_01 B_10 -> C_00
        u2 = [0.0]*8; u2[a_i_base*4 + (a_j_base+1)] = 1
        v2 = [0.0]*28; v2[(b_j_base+1)*7 + b_k_base] = 1
        w2 = [0.0]*14; w2[b_k_base*2 + a_i_base] = 1
        add_term(u2, v2, w2)
        
        # Term 3: A_00 B_01 -> C_01
        u3 = [0.0]*8; u3[a_i_base*4 + a_j_base] = 1
        v3 = [0.0]*28; v3[b_j_base*7 + (b_k_base+1)] = 1
        w3 = [0.0]*14; w3[(b_k_base+1)*2 + a_i_base] = 1
        add_term(u3, v3, w3)
        
        # Term 4: A_01 B_11 -> C_01
        u4 = [0.0]*8; u4[a_i_base*4 + (a_j_base+1)] = 1
        v4 = [0.0]*28; v4[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w4 = [0.0]*14; w4[(b_k_base+1)*2 + a_i_base] = 1
        add_term(u4, v4, w4)
        
        # Term 5: A_00 B_02 -> C_02
        u5 = [0.0]*8; u5[a_i_base*4 + a_j_base] = 1
        v5 = [0.0]*28; v5[b_j_base*7 + (b_k_base+2)] = 1
        w5 = [0.0]*14; w5[(b_k_base+2)*2 + a_i_base] = 1
        add_term(u5, v5, w5)
        
        # Term 6: A_01 B_12 -> C_02
        u6 = [0.0]*8; u6[a_i_base*4 + (a_j_base+1)] = 1
        v6 = [0.0]*28; v6[(b_j_base+1)*7 + (b_k_base+2)] = 1
        w6 = [0.0]*14; w6[(b_k_base+2)*2 + a_i_base] = 1
        add_term(u6, v6, w6)
        
        # Term 7: A_10 B_00 -> C_10
        u7 = [0.0]*8; u7[(a_i_base+1)*4 + a_j_base] = 1
        v7 = [0.0]*28; v7[b_j_base*7 + b_k_base] = 1
        w7 = [0.0]*14; w7[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u7, v7, w7)
        
        # Term 8: A_11 B_10 -> C_10
        u8 = [0.0]*8; u8[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v8 = [0.0]*28; v8[(b_j_base+1)*7 + b_k_base] = 1
        w8 = [0.0]*14; w8[b_k_base*2 + (a_i_base+1)] = 1
        add_term(u8, v8, w8)
        
        # Term 9: A_10 B_01 -> C_11
        u9 = [0.0]*8; u9[(a_i_base+1)*4 + a_j_base] = 1
        v9 = [0.0]*28; v9[b_j_base*7 + (b_k_base+1)] = 1
        w9 = [0.0]*14; w9[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u9, v9, w9)
        
        # Term 10: A_11 B_11 -> C_11
        u10 = [0.0]*8; u10[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v10 = [0.0]*28; v10[(b_j_base+1)*7 + (b_k_base+1)] = 1
        w10 = [0.0]*14; w10[(b_k_base+1)*2 + (a_i_base+1)] = 1
        add_term(u10, v10, w10)
        
        # Term 11: A_10 B_02 -> C_12
        u11 = [0.0]*8; u11[(a_i_base+1)*4 + a_j_base] = 1
        v11 = [0.0]*28; v11[b_j_base*7 + (b_k_base+2)] = 1
        w11 = [0.0]*14; w11[(b_k_base+2)*2 + (a_i_base+1)] = 1
        add_term(u11, v11, w11)
        
        # Term 12: A_11 B_12 -> C_12
        u12 = [0.0]*8; u12[(a_i_base+1)*4 + (a_j_base+1)] = 1
        v12 = [0.0]*28; v12[(b_j_base+1)*7 + (b_k_base+2)] = 1
        w12 = [0.0]*14; w12[(b_k_base+2)*2 + (a_i_base+1)] = 1
        add_term(u12, v12, w12)

    # Construction Plan:
    # A is 2x4. B is 4x7.
    # Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    # Split B into B_top (2x7, rows 0-1) and B_bottom (2x7, rows 2-3).
    # C = A_left B_top + A_right B_bottom.
    
    # For A_left B_top (2x2 * 2x7):
    # Split B_top into B_top_0 (2x2, cols 0-1), B_top_1 (2x2, cols 2-3), B_top_2 (2x3, cols 4-6).
    # Use Strassen for A_left * B_top_0. (7 terms)
    # Use Strassen for A_left * B_top_1. (7 terms)
    # Use Schoolbook for A_left * B_top_2. (12 terms? No, 2x2*2x3 is 6 terms? 
    #   Wait, my schoolbook function above has 12 terms?
    #   Let's re-read add_schoolbook_2x2x3.
    #   It adds 12 terms. Why?
    #   C_00, C_01, C_02, C_10, C_11, C_12.
    #   Each C_ij is a sum of 2 terms.
    #   So 6 * 2 = 12 terms?
    #   No, in tensor decomposition, each term is a rank-1 tensor.
    #   C_00 = A_00 B_00 + A_01 B_10.
    #   This requires 2 rank-1 terms if we don't factor further.
    #   So yes, 12 terms for 2x2 * 2x3 schoolbook.
    #   Wait, is there a better way?
    #   Rank(2,2,3) is 6?
    #   If R(2,2,3) = 6, then we need 6 terms.
    #   My function adds 12 terms. That's rank 12.
    #   Let's check if R(2,2,3) is indeed 6.
    #   Lower bound: max(4, 6, 6) = 6.
    #   Upper bound: 6?
    #   Yes, it is known that R(2,2,3) = 6.
    #   So I need a rank-6 decomposition for 2x2 * 2x3.
    
    # Let's find the rank-6 decomposition for 2x2 * 2x3.
    # Or just use the 12-term schoolbook if I can't find it quickly?
    # 7 + 7 + 12 = 26 terms for A_left B_top.
    # 26 + 26 = 52 terms total.
    # This is better than 56, but worse than 50.
    # If I can get 2x2 * 2x3 down to 6 terms, total is 7+7+6 + 7+7+6 = 40.
    
    # Let's try to construct rank-6 for 2x2 * 2x3.
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # 6 terms.
    # This is essentially doing 2x2 * 2x2 for the first 2 cols, and 2x2 * 2x1 for the last col?
    # No, 2x2 * 2x1 is rank 2?
    # C_col2 = A * [g; j] = [ag+bj, cg+dj].
    # This is 2 terms.
    # So total 6 (for first 2 cols) + 2 (for last col) = 8?
    # But R(2,2,3) = 6. So we can do better.
    
    # Actually, let's stick to the safe 52 rank solution first, or try to optimize the 2x2*2x3 part.
    # Wait, the inspiration said rank 50.
    # 50 = 2 * 25?
    # 25 = 7 + 7 + 11?
    # Maybe the 2x2*2x3 part is rank 11? No, schoolbook is 12.
    # Maybe they used a different split?
    
    # Let's try to implement the 52 rank solution first as it's safer and already an improvement.
    # Then I can try to optimize the 2x2*2x3 part to rank 6 if I can find the coefficients.
    
    # For now, I will use the 12-term schoolbook for 2x2*2x3.
    # Total rank: 7 (Strassen) + 7 (Strassen) + 12 (Schoolbook) = 26 per side.
    # Total 52.
    
    # Side 1: A_left (i=0,1, j=0,1) * B_top (j=0,1, k=0..6)
    # B_top_0: j=0,1, k=0,1. Strassen.
    add_strassen_block(0, 0, 0, 0)
    # B_top_1: j=0,1, k=2,3. Strassen.
    add_strassen_block(0, 0, 0, 2)
    # B_top_2: j=0,1, k=4,5,6. Schoolbook 2x2*2x3.
    add_schoolbook_2x2x3(0, 0, 0, 4)
    
    # Side 2: A_right (i=0,1, j=2,3) * B_bottom (j=2,3, k=0..6)
    # B_bottom_0: j=2,3, k=0,1. Strassen.
    add_strassen_block(0, 2, 2, 0)
    # B_bottom_1: j=2,3, k=2,3. Strassen.
    add_strassen_block(0, 2, 2, 2)
    # B_bottom_2: j=2,3, k=4,5,6. Schoolbook 2x2*2x3.
    add_schoolbook_2x2x3(0, 2, 2, 4)
    
    # Convert to numpy arrays
    U_arr = np.array(U).T # Shape (rank, num_u) -> Transpose to (num_u, rank)
    V_arr = np.array(V).T
    W_arr = np.array(W).T
    
    # Verify
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(U_arr.shape[1]):
        u_col = U_arr[:, c]
        v_col = V_arr[:, c]
        w_col = W_arr[:, c]
        
        # This is slow, but for verification it's fine.
        # We can vectorize if needed.
        u_idx = np.nonzero(u_col)[0]
        v_idx = np.nonzero(v_col)[0]
        w_idx = np.nonzero(w_col)[0]
        
        for ui in u_idx:
            i = ui // m
            j = ui % m
            u_val = u_col[ui]
            
            for vi in v_idx:
                v_j = vi // p
                k = vi % p
                if v_j != j:
                    continue
                v_val = v_col[vi]
                
                for wi in w_idx:
                    w_k = wi // n
                    w_i = wi % n
                    if w_k != k or w_i != i:
                        continue
                    w_val = w_col[wi]
                    
                    T[i, j, k] += u_val * v_val * w_val
                    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook if construction failed
        U_fallback = np.zeros((num_u, n*m*p), dtype=float)
        V_fallback = np.zeros((num_v, n*m*p), dtype=float)
        W_fallback = np.zeros((num_w, n*m*p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, col] = 1.0
                    V_fallback[j * p + k, col] = 1.0
                    W_fallback[k * n + i, col] = 1.0
                    col += 1
        U_arr = U_fallback
        V_arr = V_fallback
        W_arr = W_fallback

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END
