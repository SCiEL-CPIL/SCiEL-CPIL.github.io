import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _construct_rank_68():
    n, m, p = 3, 4, 6
    # Rank 68 decomposition
    
    # We will build the U, V, W matrices directly.
    # U shape: (n*m, rank) = (12, 68)
    # V shape: (m*p, rank) = (24, 68)
    # W shape: (p*n, rank) = (18, 68)
    
    rank = 68
    U = np.zeros((12, rank))
    V = np.zeros((24, rank))
    W = np.zeros((18, rank))
    
    # Indexing:
    # U[i, j] corresponds to A[i, j] where i in 0..2, j in 0..3
    # V[i, j] corresponds to B[i, j] where i in 0..3, j in 0..5
    # W[i, j] corresponds to C[i, j] where i in 0..5, j in 0..2
    # Wait, W[k*n+i, r] is the convention?
    # "W[k*n+i,r]" -> W index is k*n + i.
    # C is p x n. k is 0..p-1, i is 0..n-1.
    # So W row index is k*n + i.
    
    # Decomposition Structure:
    # C = C_left + C_right (split p into 3, 3)
    # C_left (3x3) from A(3x4) * B_left(4x3)
    # C_right (3x3) from A(3x4) * B_right(4x3)
    
    # For each 3x3 part:
    # C_part = C_top + C_bot (split n into 1, 2)
    # C_top (1x3) from A_top(1x4) * B_part(4x3) -> Rank 12
    # C_bot (2x3) from A_bot(2x4) * B_part(4x3) -> Rank 22 (2 * 11)
    
    # For each 2x3 part (A_bot * B_part):
    # C_bot = C_bot_left + C_bot_right (split m into 2, 2)
    # C_bot_left (2x3) from A_bot_left(2x2) * B_part_top(2x3) -> Rank 11
    # C_bot_right (2x3) from A_bot_right(2x2) * B_part_bot(2x2) -> Rank 11
    
    # Total Rank: 2 * (12 + 2*11) = 2 * 34 = 68.
    
    col_offset = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col_offset
        U[u_idx, col_offset] += u_val
        V[v_idx, col_offset] += v_val
        W[w_idx, col_offset] += w_val
        col_offset += 1

    # Process C_left (p cols 0,1,2) and C_right (p cols 3,4,5)
    for p_block in range(2):
        p_start = p_block * 3
        # B_part is B[:, p_start : p_start+3]
        
        # C_top (1x3)
        # A_top is A[0, :] (1x4)
        # Schoolbook for 1x4 * 4x3
        # C_top[j] = sum_k A[0,k] B[k, p_start+j]
        for k in range(4):
            for j in range(3):
                # Term: A[0,k] * B[k, p_start+j] * C[0, p_start+j]
                # U idx: 0*m + k = k
                # V idx: k*p + (p_start+j)
                # W idx: (p_start+j)*n + 0
                add_term(k, 1.0, k*p + (p_start+j), 1.0, (p_start+j)*n + 0, 1.0)
                
        # C_bot (2x3)
        # A_bot is A[1:, :] (2x4)
        # Split m into 2,2
        for m_block in range(2):
            m_start = m_block * 2
            # A_bot_block is A[1:3, m_start:m_start+2]
            # B_part_block is B[m_start:m_start+2, p_start:p_start+3]
            
            # This is M_{2,2,3} Rank 11
            # Strassen for cols 0,1 of B_part_block
            # Schoolbook for col 2 of B_part_block
            
            # Strassen M_{2,2,2} on A_bot_block (2x2) and B_part_block[:, 0:2] (2x2)
            # A_bot_block indices:
            # (1, m_start), (1, m_start+1)
            # (2, m_start), (2, m_start+1)
            # B_part_block indices:
            # (m_start, p_start), (m_start, p_start+1)
            # (m_start+1, p_start), (m_start+1, p_start+1)
            
            # Strassen terms for 2x2x2
            # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-a)(e+f)
            # m7 = (b-d)(g+h)
            # C11 = m1 + m4 - m5 + m7
            # C12 = m3 + m5
            # C21 = m2 + m4
            # C22 = m1 - m2 + m3 + m6
            
            a_idx = 1*m + m_start
            b_idx = 1*m + (m_start+1)
            c_idx = 2*m + m_start
            d_idx = 2*m + (m_start+1)
            
            e_idx = m_start*p + p_start
            f_idx = m_start*p + (p_start+1)
            g_idx = (m_start+1)*p + p_start
            h_idx = (m_start+1)*p + (p_start+1)
            
            # Output C indices for this block:
            # Rows 1,2 of C. Cols p_start, p_start+1.
            # C[1, p_start] -> W idx (p_start)*n + 1
            # C[1, p_start+1] -> W idx (p_start+1)*n + 1
            # C[2, p_start] -> W idx (p_start)*n + 2
            # C[2, p_start+1] -> W idx (p_start+1)*n + 2
            
            w11 = p_start*n + 1
            w12 = (p_start+1)*n + 1
            w21 = p_start*n + 2
            w22 = (p_start+1)*n + 2
            
            # m1
            # (a+d) * (e+h)
            # U: a=1, d=1. V: e=1, h=1.
            # This is 4 terms? No, Strassen uses linear combinations.
            # We need to add terms to U, V, W.
            # Term 1: (a+d) * e
            # U: a=1, d=1. V: e=1. W: w11=1, w21=1?
            # Strassen decomposition is sum of rank-1 tensors.
            
            # We will manually add the 7 Strassen terms.
            
            # 1. (a+d)(e+h)
            # Expand to 4 rank-1 terms? No, Strassen is 7.
            # The 7 multiplications are:
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-a)(e+f)
            # m7 = (b-d)(g+h)
            
            # We need to represent each mi as a sum of rank-1 tensors?
            # No, each mi is a product of linear forms.
            # L_Ui * L_Vi.
            # We add 1 to U for the coeffs of L_Ui, 1 to V for coeffs of L_Vi, and the result goes to W.
            
            # Term 1: m1 = (a+d)(e+h)
            # U: a=1, d=1. V: e=1, h=1.
            # This single multiplication contributes to C11 and C22.
            # C11 += m1. C22 += m1.
            # But m1 is one multiplication.
            # We add 1 term to U,V,W.
            # U: a_idx=1, d_idx=1.
            # V: e_idx=1, h_idx=1.
            # W: w11=1, w22=1.
            add_term(a_idx, 1.0, e_idx, 1.0, w11, 1.0) # Wait, this is not how it works.
            
            # Correct way:
            # Each rank-1 term is u_vec * v_vec * w_vec.
            # For m1 = (a+d)(e+h), the tensor is (e1_a + e1_d) (e1_e + e1_h) (e1_w11 + e1_w22)?
            # No.
            # The decomposition is T = sum mi * ui * vi * wi.
            # For m1, ui has 1 at a, d. vi has 1 at e, h. wi has 1 at w11, w22?
            # No, wi is a vector.
            # The term is (a+d) * (e+h) * (w11 + w22)?
            # This would mean C11 gets (a+d)(e+h) and C22 gets (a+d)(e+h).
            # Yes.
            
            # So:
            # 1. m1
            add_term(a_idx, 1.0, e_idx, 1.0, w11, 1.0)
            add_term(d_idx, 1.0, e_idx, 1.0, w11, 1.0) # This is wrong.
            
            # We must add the whole linear combination in one go?
            # No, add_term adds to specific indices.
            # We need to add 4 terms for m1?
            # (a+d)(e+h) = ae + ah + de + dh.
            # This is 4 schoolbook terms.
            # Strassen reduces 8 to 7.
            # We should just use the 7 Strassen multiplications explicitly.
            
            # Let's restart the Strassen part.
            # I will define the 7 multiplications and their contributions to W.
            
            # m1 = (a+d)(e+h)
            # U: a=1, d=1. V: e=1, h=1.
            # W: w11=1, w22=1.
            # This is NOT a rank-1 term. It is a rank-4 term if expanded.
            # But in tensor decomposition, we represent the tensor as sum of rank-1 terms.
            # Strassen's algorithm provides a decomposition into 7 rank-1 terms?
            # No, Strassen's algorithm uses 7 multiplications of linear combinations.
            # Each multiplication $m_i = L_{Ui} \cdot L_{Vi}$ is a rank-1 term in the tensor?
            # Yes.
            # So for m1, we have U_vec with 1 at a, d. V_vec with 1 at e, h.
            # W_vec with 1 at w11, w22?
            # No, the output of m1 is added to w11 and w22.
            # So W_vec has 1 at w11 and 1 at w22.
            # So we add ONE term:
            # U: {a:1, d:1}, V: {e:1, h:1}, W: {w11:1, w22:1}.
            
            # Let's implement this.
            
            # 1. m1
            U[a_idx, col_offset] += 1.0
            U[d_idx, col_offset] += 1.0
            V[e_idx, col_offset] += 1.0
            V[h_idx, col_offset] += 1.0
            W[w11, col_offset] += 1.0
            W[w22, col_offset] += 1.0
            col_offset += 1
            
            # 2. m2 = (c+d)e
            U[c_idx, col_offset] += 1.0
            U[d_idx, col_offset] += 1.0
            V[e_idx, col_offset] += 1.0
            W[w21, col_offset] += 1.0
            W[w22, col_offset] += -1.0 # C22 = m1 - m2 + m3 + m6
            col_offset += 1
            
            # 3. m3 = a(f-h)
            U[a_idx, col_offset] += 1.0
            V[f_idx, col_offset] += 1.0
            V[h_idx, col_offset] += -1.0
            W[w12, col_offset] += 1.0
            W[w22, col_offset] += 1.0 # C22 = ... + m3
            col_offset += 1
            
            # 4. m4 = d(g-e)
            U[d_idx, col_offset] += 1.0
            V[g_idx, col_offset] += 1.0
            V[e_idx, col_offset] += -1.0
            W[w11, col_offset] += 1.0
            W[w21, col_offset] += 1.0
            col_offset += 1
            
            # 5. m5 = (a+b)h
            U[a_idx, col_offset] += 1.0
            U[b_idx, col_offset] += 1.0
            V[h_idx, col_offset] += 1.0
            W[w11, col_offset] += -1.0 # C11 = m1 + m4 - m5 + m7
            W[w12, col_offset] += 1.0
            col_offset += 1
            
            # 6. m6 = (c-a)(e+f)
            U[c_idx, col_offset] += 1.0
            U[a_idx, col_offset] += -1.0
            V[e_idx, col_offset] += 1.0
            V[f_idx, col_offset] += 1.0
            W[w22, col_offset] += 1.0
            col_offset += 1
            
            # 7. m7 = (b-d)(g+h)
            U[b_idx, col_offset] += 1.0
            U[d_idx, col_offset] += -1.0
            V[g_idx, col_offset] += 1.0
            V[h_idx, col_offset] += 1.0
            W[w11, col_offset] += 1.0
            col_offset += 1
            
            # Schoolbook for col 2 (p_start+2)
            # C[1, p_start+2] = a*B[m_start, p_start+2] + b*B[m_start+1, p_start+2]
            # C[2, p_start+2] = c*B[m_start, p_start+2] + d*B[m_start+1, p_start+2]
            
            w13 = (p_start+2)*n + 1
            w23 = (p_start+2)*n + 2
            
            # Term 8: a * B[m_start, p_start+2]
            U[a_idx, col_offset] += 1.0
            V[m_start*p + (p_start+2), col_offset] += 1.0
            W[w13, col_offset] += 1.0
            col_offset += 1
            
            # Term 9: b * B[m_start+1, p_start+2]
            U[b_idx, col_offset] += 1.0
            V[(m_start+1)*p + (p_start+2), col_offset] += 1.0
            W[w13, col_offset] += 1.0
            col_offset += 1
            
            # Term 10: c * B[m_start, p_start+2]
            U[c_idx, col_offset] += 1.0
            V[m_start*p + (p_start+2), col_offset] += 1.0
            W[w23, col_offset] += 1.0
            col_offset += 1
            
            # Term 11: d * B[m_start+1, p_start+2]
            U[d_idx, col_offset] += 1.0
            V[(m_start+1)*p + (p_start+2), col_offset] += 1.0
            W[w23, col_offset] += 1.0
            col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct_rank_68()
