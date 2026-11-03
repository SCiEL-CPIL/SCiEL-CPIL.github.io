"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank-7 decomposition for 2x2 matrices
    # U2, V2, W2 shapes (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Factor 0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Factor 1: M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # Factor 2: M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # Factor 3: M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = -1; W2[2][3] = 1
    
    # Factor 4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    
    # Factor 5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Factor 6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1
    
    # Construct 4x4 decomposition using Kronecker product of the 2x2 decomposition
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all combinations of outer and inner indices
    for i_out in range(2):
        for j_out in range(2):
            for k_out in range(2):
                for c_out in range(7):
                    for i_in in range(2):
                        for j_in in range(2):
                            for k_in in range(2):
                                for c_in in range(7):
                                    col = c_out * 7 + c_in
                                    
                                    # Calculate row indices for 4x4 matrices
                                    # U index: i*m + j = i*4 + j
                                    # i = i_out*2 + i_in, j = j_out*2 + j_in
                                    u_row = (i_out * 2 + i_in) * 4 + (j_out * 2 + j_in)
                                    # V index: j*p + k = j*4 + k
                                    # j = j_out*2 + j_in, k = k_out*2 + k_in
                                    v_row = (j_out * 2 + j_in) * 4 + (k_out * 2 + k_in)
                                    # W index: k*n + i = k*4 + i
                                    # k = k_out*2 + k_in, i = i_out*2 + i_in
                                    w_row = (k_out * 2 + k_in) * 4 + (i_out * 2 + i_in)
                                    
                                    # Get values from 2x2 decomposition
                                    u_val_out = U2[i_out * 2 + j_out][c_out]
                                    u_val_in = U2[i_in * 2 + j_in][c_in]
                                    v_val_out = V2[j_out * 2 + k_out][c_out]
                                    v_val_in = V2[j_in * 2 + k_in][c_in]
                                    w_val_out = W2[k_out * 2 + i_out][c_out]
                                    w_val_in = W2[k_in * 2 + i_in][c_in]
                                    
                                    u_val = u_val_out * u_val_in
                                    v_val = v_val_out * v_val_in
                                    w_val = w_val_out * w_val_in
                                    
                                    U[u_row][col] = u_val
                                    V[v_row][col] = v_val
                                    W[w_row][col] = w_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
