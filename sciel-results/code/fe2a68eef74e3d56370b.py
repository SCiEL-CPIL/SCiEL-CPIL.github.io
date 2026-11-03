"""Matrix multiplication 6x6x6 using 2x2 block decomposition with Strassen + schoolbook 3x3."""
import time as _time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-189 tensor decomposition for 6x6 matrix multiplication.
    Strategy: Partition into 2x2 blocks of 3x3 matrices, apply Strassen's 7-multiplication
    algorithm at the block level, and use the schoolbook (rank-27) algorithm for each 3x3 multiplication.
    Total rank = 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    
    # U shape: (n*m, rank) = (36, 189)
    # V shape: (m*p, rank) = (36, 189)
    # W shape: (p*n, rank) = (36, 189)
    # Note: W[k*n+i, r] is the official trace convention.
    # Standard tensor equation: C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication.
    # We decompose A, B, C into 2x2 blocks of size 3x3.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 products are:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Define coefficients for A, B, C blocks in each product.
    # Block indices: (row, col) in {0, 1}
    
    # A coefficients: beta[s][I][J]
    # B coefficients: gamma[s][K][L]
    # C coefficients: alpha[s][I][K]
    
    beta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    alpha = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # s=0: M0
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    # s=1: M1
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    # s=2: M2
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    # s=3: M3
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    # s=4: M4
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    # s=5: M5
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    # s=6: M6
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1
    
    # Loop over Strassen steps and 3x3 schoolbook indices
    # Schoolbook for 3x3:
    # For i, j, k in 0..2:
    #   U3[i*3+j, col] = 1
    #   V3[j*3+k, col] = 1
    #   W3[k*3+i, col] = 1
    
    block_size = 3
    
    for s in range(7):
        base_col = s * 27
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = base_col + i_l * 9 + j_l * 3 + k_l
                    
                    # Update U
                    # U row index: global_i * m + global_j
                    # global_i = block_row * block_size + i_l
                    # global_j = block_col * block_size + j_l
                    for I in range(2):
                        for J in range(2):
                            coeff = beta[s][I][J]
                            if coeff != 0:
                                global_i = I * block_size + i_l
                                global_j = J * block_size + j_l
                                u_idx = global_i * m + global_j
                                U[u_idx][col] += coeff
                    
                    # Update V
                    # V row index: global_j * p + global_k
                    # global_j = block_row * block_size + j_l
                    # global_k = block_col * block_size + k_l
                    for K in range(2):
                        for L in range(2):
                            coeff = gamma[s][K][L]
                            if coeff != 0:
                                global_j = K * block_size + j_l
                                global_k = L * block_size + k_l
                                v_idx = global_j * p + global_k
                                V[v_idx][col] += coeff
                    
                    # Update W
                    # W row index: global_k * n + global_i
                    # global_k = block_col * block_size + k_l
                    # global_i = block_row * block_size + i_l
                    for I in range(2):
                        for K in range(2):
                            coeff = alpha[s][I][K]
                            if coeff != 0:
                                global_i = I * block_size + i_l
                                global_k = K * block_size + k_l
                                w_idx = global_k * n + global_i
                                W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
