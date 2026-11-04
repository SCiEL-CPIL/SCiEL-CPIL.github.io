import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    # Total rank: 20 + 16 + 49 = 85
    rank = 85
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    
    # Block 1: a4 b4^T (Rank 20)
    # A[:, 4] * B[4, :]
    # C[i, k] += A[i, 4] * B[4, k]
    for i in range(4):
        for k in range(5):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            
    # Block 2: A' B''' (Rank 16)
    # A[:, 0:4] * B[0:4, 4]
    # C[i, 4] += sum_j A[i, j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    # Block 3: A' B'' (Rank 49)
    # A[:, 0:4] * B[0:4, 0:4]
    # C[0:4, 0:4] += A' B''
    
    # Inner Strassen terms for 2x2x2
    # t_m: (X_coeffs, Y_coeffs, Z_coeffs)
    # X, Y, Z are 2x2 matrices flattened to length 4 [00, 01, 10, 11]
    inner_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # t1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # t2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # t3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]), # t4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # t5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # t6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # t7
    ]
    
    # Outer Strassen terms for 4x4x4
    # Each term l defines X_l, Y_l, Z_l
    # X_l[a,b] is a linear combo of A[i,j]
    # We represent alpha_l as a list of 4 dicts or 4x4 arrays
    # Let's use 4x4 numpy arrays for alpha_l[a,b], beta_l[a,b], gamma_l[a,b]
    
    # Initialize alpha, beta, gamma for the 7 outer terms
    # alpha[l][a][b] is a 4x4 matrix
    # This is too big to store statically? 7 * 2 * 2 * 4 * 4 = 448 floats. Small.
    
    # Let's define the mappings.
    # A' is A[:, 0:4]. Indices i, j in 0..3.
    # B'' is B[0:4, 0:4]. Indices j, k in 0..3.
    # C' is C[0:4, 0:4]. Indices i, k in 0..3.
    
    # Helper to create a 4x4 zero matrix
    def z4():
        return np.zeros((4, 4), dtype=np.float64)
    
    # M1 = (A00 + A11)(B00 + B11)
    # A00 = A[0:2, 0:2], A11 = A[2:4, 2:4]
    # X1[0,0] = A[0,0] + A[2,2]
    # X1[0,1] = A[0,1] + A[2,3]
    # X1[1,0] = A[1,0] + A[3,2]
    # X1[1,1] = A[1,1] + A[3,3]
    
    def add_block_to_alpha(alpha, a, b, i_start, j_start, sign=1):
        for di in range(2):
            for dj in range(2):
                alpha[a][b][i_start+di][j_start+dj] += sign

    # We need to construct alpha_l, beta_l, gamma_l for l=0..6
    alphas = [ [ [z4() for _ in range(2)] for _ in range(2) ] for _ in range(7) ]
    betas = [ [ [z4() for _ in range(2)] for _ in range(2) ] for _ in range(7) ]
    gammas = [ [ [z4() for _ in range(2)] for _ in range(2) ] for _ in range(7) ]
    
    # Term 0: M1
    # X1 = A00 + A11
    add_block_to_alpha(alphas[0], 0, 0, 0, 0, 1)
    add_block_to_alpha(alphas[0], 0, 1, 0, 2, 1)
    add_block_to_alpha(alphas[0], 1, 0, 2, 0, 1)
    add_block_to_alpha(alphas[0], 1, 1, 2, 2, 1)
    # Y1 = B00 + B11
    add_block_to_alpha(betas[0], 0, 0, 0, 0, 1)
    add_block_to_alpha(betas[0], 0, 1, 0, 2, 1)
    add_block_to_alpha(betas[0], 1, 0, 2, 0, 1)
    add_block_to_alpha(betas[0], 1, 1, 2, 2, 1)
    # Z1 contributes to C00 (top-left 2x2 of C')
    # C00 is C[0:2, 0:2]
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[0][a][b][a*2+di][b*2+dj] = 1
                    
    # Term 1: M2 = (A10 + A11) B00
    # X2[0,0] = A[2,0] + A[0,0]? No.
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # X2 = A10 + A11
    # X2[0,0] = A[2,0] + A[2,2]
    # X2[0,1] = A[2,1] + A[2,3]
    # X2[1,0] = A[3,0] + A[3,2]
    # X2[1,1] = A[3,1] + A[3,3]
    add_block_to_alpha(alphas[1], 0, 0, 2, 0, 1)
    add_block_to_alpha(alphas[1], 0, 1, 2, 2, 1)
    add_block_to_alpha(alphas[1], 1, 0, 3, 0, 1)
    add_block_to_alpha(alphas[1], 1, 1, 3, 2, 1)
    # Y2 = B00
    add_block_to_alpha(betas[1], 0, 0, 0, 0, 1)
    add_block_to_alpha(betas[1], 0, 1, 0, 1, 1)
    add_block_to_alpha(betas[1], 1, 0, 1, 0, 1)
    add_block_to_alpha(betas[1], 1, 1, 1, 1, 1)
    # Z2 contributes to C10 (bottom-left 2x2 of C')
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[1][a][b][2+a*2+di][b*2+dj] = 1
                    
    # Term 2: M3 = A00 (B10 - B11)
    # X3 = A00
    add_block_to_alpha(alphas[2], 0, 0, 0, 0, 1)
    add_block_to_alpha(alphas[2], 0, 1, 0, 1, 1)
    add_block_to_alpha(alphas[2], 1, 0, 1, 0, 1)
    add_block_to_alpha(alphas[2], 1, 1, 1, 1, 1)
    # Y3 = B10 - B11
    add_block_to_alpha(betas[2], 0, 0, 2, 0, 1)
    add_block_to_alpha(betas[2], 0, 1, 2, 1, 1)
    add_block_to_alpha(betas[2], 1, 0, 3, 0, 1)
    add_block_to_alpha(betas[2], 1, 1, 3, 1, 1)
    add_block_to_alpha(betas[2], 0, 0, 2, 2, -1)
    add_block_to_alpha(betas[2], 0, 1, 2, 3, -1)
    add_block_to_alpha(betas[2], 1, 0, 3, 2, -1)
    add_block_to_alpha(betas[2], 1, 1, 3, 3, -1)
    # Z3 contributes to C01 (top-right 2x2 of C')
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[2][a][b][a*2+di][2+b*2+dj] = 1
                    
    # Term 3: M4 = A11 (B01 - B00)
    # X4 = A11
    add_block_to_alpha(alphas[3], 0, 0, 2, 2, 1)
    add_block_to_alpha(alphas[3], 0, 1, 2, 3, 1)
    add_block_to_alpha(alphas[3], 1, 0, 3, 2, 1)
    add_block_to_alpha(alphas[3], 1, 1, 3, 3, 1)
    # Y4 = B01 - B00
    add_block_to_alpha(betas[3], 0, 0, 0, 2, 1)
    add_block_to_alpha(betas[3], 0, 1, 0, 3, 1)
    add_block_to_alpha(betas[3], 1, 0, 1, 2, 1)
    add_block_to_alpha(betas[3], 1, 1, 1, 3, 1)
    add_block_to_alpha(betas[3], 0, 0, 0, 0, -1)
    add_block_to_alpha(betas[3], 0, 1, 0, 1, -1)
    add_block_to_alpha(betas[3], 1, 0, 1, 0, -1)
    add_block_to_alpha(betas[3], 1, 1, 1, 1, -1)
    # Z4 contributes to C00 and C10
    # C00 += M4, C10 += M4
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[3][a][b][a*2+di][b*2+dj] = 1
                    gammas[3][a][b][2+a*2+di][b*2+dj] = 1
                    
    # Term 4: M5 = (A00 + A01) B11
    # X5 = A00 + A01
    add_block_to_alpha(alphas[4], 0, 0, 0, 0, 1)
    add_block_to_alpha(alphas[4], 0, 1, 0, 1, 1)
    add_block_to_alpha(alphas[4], 1, 0, 1, 0, 1)
    add_block_to_alpha(alphas[4], 1, 1, 1, 1, 1)
    add_block_to_alpha(alphas[4], 0, 0, 0, 2, 1)
    add_block_to_alpha(alphas[4], 0, 1, 0, 3, 1)
    add_block_to_alpha(alphas[4], 1, 0, 1, 2, 1)
    add_block_to_alpha(alphas[4], 1, 1, 1, 3, 1)
    # Y5 = B11
    add_block_to_alpha(betas[4], 0, 0, 2, 2, 1)
    add_block_to_alpha(betas[4], 0, 1, 2, 3, 1)
    add_block_to_alpha(betas[4], 1, 0, 3, 2, 1)
    add_block_to_alpha(betas[4], 1, 1, 3, 3, 1)
    # Z5 contributes to -C00 and C01
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[4][a][b][a*2+di][b*2+dj] = -1
                    gammas[4][a][b][a*2+di][2+b*2+dj] = 1
                    
    # Term 5: M6 = (A10 - A00) (B00 + B01)
    # X6 = A10 - A00
    add_block_to_alpha(alphas[5], 0, 0, 2, 0, 1)
    add_block_to_alpha(alphas[5], 0, 1, 2, 1, 1)
    add_block_to_alpha(alphas[5], 1, 0, 3, 0, 1)
    add_block_to_alpha(alphas[5], 1, 1, 3, 1, 1)
    add_block_to_alpha(alphas[5], 0, 0, 0, 0, -1)
    add_block_to_alpha(alphas[5], 0, 1, 0, 1, -1)
    add_block_to_alpha(alphas[5], 1, 0, 1, 0, -1)
    add_block_to_alpha(alphas[5], 1, 1, 1, 1, -1)
    # Y6 = B00 + B01
    add_block_to_alpha(betas[5], 0, 0, 0, 0, 1)
    add_block_to_alpha(betas[5], 0, 1, 0, 1, 1)
    add_block_to_alpha(betas[5], 1, 0, 1, 0, 1)
    add_block_to_alpha(betas[5], 1, 1, 1, 1, 1)
    add_block_to_alpha(betas[5], 0, 0, 0, 2, 1)
    add_block_to_alpha(betas[5], 0, 1, 0, 3, 1)
    add_block_to_alpha(betas[5], 1, 0, 1, 2, 1)
    add_block_to_alpha(betas[5], 1, 1, 1, 3, 1)
    # Z6 contributes to C11
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[5][a][b][2+a*2+di][2+b*2+dj] = 1
                    
    # Term 6: M7 = (A01 - A11) (B10 + B11)
    # X7 = A01 - A11
    add_block_to_alpha(alphas[6], 0, 0, 0, 2, 1)
    add_block_to_alpha(alphas[6], 0, 1, 0, 3, 1)
    add_block_to_alpha(alphas[6], 1, 0, 1, 2, 1)
    add_block_to_alpha(alphas[6], 1, 1, 1, 3, 1)
    add_block_to_alpha(alphas[6], 0, 0, 2, 2, -1)
    add_block_to_alpha(alphas[6], 0, 1, 2, 3, -1)
    add_block_to_alpha(alphas[6], 1, 0, 3, 2, -1)
    add_block_to_alpha(alphas[6], 1, 1, 3, 3, -1)
    # Y7 = B10 + B11
    add_block_to_alpha(betas[6], 0, 0, 2, 0, 1)
    add_block_to_alpha(betas[6], 0, 1, 2, 1, 1)
    add_block_to_alpha(betas[6], 1, 0, 3, 0, 1)
    add_block_to_alpha(betas[6], 1, 1, 3, 1, 1)
    add_block_to_alpha(betas[6], 0, 0, 2, 2, 1)
    add_block_to_alpha(betas[6], 0, 1, 2, 3, 1)
    add_block_to_alpha(betas[6], 1, 0, 3, 2, 1)
    add_block_to_alpha(betas[6], 1, 1, 3, 3, 1)
    # Z7 contributes to C00
    for a in range(2):
        for b in range(2):
            for di in range(2):
                for dj in range(2):
                    gammas[6][a][b][a*2+di][b*2+dj] = 1

    # Now combine outer and inner terms
    for l in range(7):
        for m in range(7):
            s_coeff = np.array(inner_terms[m][0])
            t_coeff = np.array(inner_terms[m][1])
            u_coeff = np.array(inner_terms[m][2])
            
            # Construct U_term, V_term, W_term
            U_term = np.zeros((nm,), dtype=np.float64)
            V_term = np.zeros((mp,), dtype=np.float64)
            W_term = np.zeros((pn,), dtype=np.float64)
            
            for a in range(2):
                for b in range(2):
                    # X_l[a,b] contribution to U
                    U_contrib = s_coeff[a*2+b] * alphas[l][a][b]
                    # Y_l[a,b] contribution to V
                    V_contrib = t_coeff[a*2+b] * betas[l][a][b]
                    # Z_l[a,b] contribution to W
                    W_contrib = u_coeff[a*2+b] * gammas[l][a][b]
                    
                    for i in range(4):
                        for j in range(4):
                            val = U_contrib[i][j]
                            if val != 0:
                                U_term[i*5 + j] += val
                    for j in range(4):
                        for k in range(4):
                            val = V_contrib[j][k]
                            if val != 0:
                                V_term[j*5 + k] += val
                    for i in range(4):
                        for k in range(4):
                            val = W_contrib[i][k]
                            if val != 0:
                                W_term[k*4 + i] += val
                                
            U[:, col] = U_term
            V[:, col] = V_term
            W[:, col] = W_term
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
