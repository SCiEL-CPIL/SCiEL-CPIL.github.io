"""Program for 6x6 matrix multiplication with rank 189 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition:
    # 7 Strassen terms for 2x2 blocks of 3x3 matrices.
    # Each 3x3 multiplication uses schoolbook rank 27.
    # Total rank = 7 * 27 = 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block matrices
    # t: term index 0..6
    # alpha[t][I][J]: coeff for A[I*3+i', J*3+j']
    # beta[t][Jp][K]: coeff for B[Jp*3+j', K*3+k']
    # gamma[t][I][K]: coeff for C[I*3+i', K*3+k']
    
    # M1 = (A00 + A11)(B00 + B11)
    alpha_0 = { (0,0): 1, (1,1): 1 }
    beta_0  = { (0,0): 1, (1,1): 1 }
    gamma_0 = { (0,0): 1, (1,1): 1 }
    
    # M2 = (A10 + A11)B00
    alpha_1 = { (1,0): 1, (1,1): 1 }
    beta_1  = { (0,0): 1 }
    gamma_1 = { (1,0): 1, (1,1): -1 }
    
    # M3 = A00(B01 - B11)
    alpha_2 = { (0,0): 1 }
    beta_2  = { (0,1): 1, (1,1): -1 }
    gamma_2 = { (0,1): 1, (1,1): 1 }
    
    # M4 = A11(B10 - B00)
    alpha_3 = { (1,1): 1 }
    beta_3  = { (1,0): 1, (0,0): -1 }
    gamma_3 = { (0,0): 1, (1,0): 1 }
    
    # M5 = (A00 + A01)B11
    alpha_4 = { (0,0): 1, (0,1): 1 }
    beta_4  = { (1,1): 1 }
    gamma_4 = { (0,0): -1, (0,1): 1 }
    
    # M6 = (A10 - A00)(B00 + B01)
    alpha_5 = { (1,0): 1, (0,0): -1 }
    beta_5  = { (0,0): 1, (0,1): 1 }
    gamma_5 = { (1,1): 1 }
    
    # M7 = (A01 - A11)(B10 + B11)
    alpha_6 = { (0,1): 1, (1,1): -1 }
    beta_6  = { (1,0): 1, (1,1): 1 }
    gamma_6 = { (0,0): 1 }
    
    alphas = [alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6]
    betas  = [beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6]
    gammas = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6]
    
    for t in range(7):
        for ip in range(3):
            for jp in range(3):
                for kp in range(3):
                    c = t * 27 + ip * 9 + jp * 3 + kp
                    
                    # Fill U
                    for (I, J), val in alphas[t].items():
                        i_global = I * 3 + ip
                        j_global = J * 3 + jp
                        U[i_global * 6 + j_global][c] = val
                        
                    # Fill V
                    for (Jp, K), val in betas[t].items():
                        j_global = Jp * 3 + jp
                        k_global = K * 3 + kp
                        V[j_global * 6 + k_global][c] = val
                        
                    # Fill W
                    for (I, K), val in gammas[t].items():
                        k_global = K * 3 + kp
                        i_global = I * 3 + ip
                        W[k_global * 6 + i_global][c] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
