"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def _schoolbook_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication, rank 27.
    U: 9x27, V: 9x27, W: 9x27
    Convention: W[k*3 + i, r]
    """
    n = 3
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 110.0
    
    n, m, p = DIMENSIONS # 6, 6, 6
    
    # Get 3x3 schoolbook decomposition
    U3, V3, W3 = _schoolbook_3x3()
    rank3 = 27
    
    # Strassen coefficients for 2x2 block structure
    # k: 1..7
    # alpha[k][I][J]: coefficient for A[I,J] in X_k
    # beta[k][I][J]: coefficient for B[I,J] in Y_k
    # gamma[k][I][J]: coefficient for M_k in C[I,J]
    
    # Initialize with zeros
    alpha = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(8)]
    beta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(8)]
    gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(8)]
    
    # k=1: M1 = (A00 + A11)(B00 + B11)
    alpha[1][0][0] = 1; alpha[1][1][1] = 1
    beta[1][0][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1; gamma[1][1][1] = 1
    
    # k=2: M2 = (A10 + A11)B00
    alpha[2][1][0] = 1; alpha[2][1][1] = 1
    beta[2][0][0] = 1
    gamma[2][1][0] = 1; gamma[2][1][1] = -1
    
    # k=3: M3 = A00(B01 - B11)
    alpha[3][0][0] = 1
    beta[3][0][1] = 1; beta[3][1][1] = -1
    gamma[3][0][1] = 1; gamma[3][1][1] = 1
    
    # k=4: M4 = A11(B10 - B00)
    alpha[4][1][1] = 1
    beta[4][1][0] = 1; beta[4][0][0] = -1
    gamma[4][0][0] = 1; gamma[4][1][0] = 1
    
    # k=5: M5 = (A00 + A01)B11
    alpha[5][0][0] = 1; alpha[5][0][1] = 1
    beta[5][1][1] = 1
    gamma[5][0][0] = -1; gamma[5][0][1] = 1
    
    # k=6: M6 = (A10 - A00)(B00 + B01)
    alpha[6][1][0] = 1; alpha[6][0][0] = -1
    beta[6][0][0] = 1; beta[6][0][1] = 1
    gamma[6][1][1] = 1
    
    # k=7: M7 = (A01 - A11)(B10 + B11)
    alpha[7][0][1] = 1; alpha[7][1][1] = -1
    beta[7][1][0] = 1; beta[7][1][1] = 1
    gamma[7][0][0] = 1
    
    rank6 = 7 * rank3
    U6 = [[0] * rank6 for _ in range(36)]
    V6 = [[0] * rank6 for _ in range(36)]
    W6 = [[0] * rank6 for _ in range(36)]
    
    # Construct U6, V6, W6
    for k in range(1, 8):
        base_col = (k - 1) * rank3
        for r in range(rank3):
            col = base_col + r
            
            # Update U6
            # U6[i_global * 6 + j_global, col]
            for I in range(2):
                for J in range(2):
                    a_val = alpha[k][I][J]
                    if a_val != 0:
                        for i_prime in range(3):
                            for j_prime in range(3):
                                i_global = I * 3 + i_prime
                                j_global = J * 3 + j_prime
                                u_idx = i_global * 6 + j_global
                                u3_idx = i_prime * 3 + j_prime
                                U6[u_idx][col] += a_val * U3[u3_idx][r]
                                
            # Update V6
            # V6[j_global * 6 + k_global, col]
            for I in range(2):
                for J in range(2):
                    b_val = beta[k][I][J]
                    if b_val != 0:
                        for j_prime in range(3):
                            for k_prime in range(3):
                                j_global = J * 3 + j_prime
                                k_global = I * 3 + k_prime
                                v_idx = j_global * 6 + k_global
                                v3_idx = j_prime * 3 + k_prime
                                V6[v_idx][col] += b_val * V3[v3_idx][r]
                                
            # Update W6
            # W6[k_global * 6 + i_global, col]
            for I in range(2):
                for J in range(2):
                    c_val = gamma[k][I][J]
                    if c_val != 0:
                        for i_prime in range(3):
                            for k_prime in range(3):
                                i_global = I * 3 + i_prime
                                k_global = J * 3 + k_prime
                                w_idx = k_global * 6 + i_global
                                w3_idx = k_prime * 3 + i_prime
                                W6[w_idx][col] += c_val * W3[w3_idx][r]
                                
    if time.time() > deadline:
        pass

    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
