"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Row index for U2: i*2 + j  (i,j in 0..1)
    # Row index for V2: j*2 + k  (j,k in 0..1)
    # Row index for W2: k*2 + i  (k,i in 0..1)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # alpha=0: M1 = (A00+A11)*(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # alpha=1: M2 = (A10+A11)*B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # alpha=2: M3 = A00*(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # alpha=3: M4 = A11*(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # alpha=4: M5 = (A00+A01)*B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # alpha=5: M6 = (A10-A00)*(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # alpha=6: M7 = (A01-A11)*(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4 matrices using Kronecker-like product
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Iterate over outer alpha (block-level) and inner beta (scalar-level)
    for alpha in range(7):
        for beta in range(7):
            t = alpha * 7 + beta
            
            # Build U: U[i*4+j][t] = U2[i1*2+j1][alpha] * U2[i2*2+j2][beta]
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    row_u = i * 4 + j
                    u_val = U2[i1*2 + j1][alpha] * U2[i2*2 + j2][beta]
                    if u_val != 0:
                        U[row_u][t] = u_val
            
            # Build V: V[j*4+k][t] = V2[j1*2+k1][alpha] * V2[j2*2+k2][beta]
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    row_v = j * 4 + k
                    v_val = V2[j1*2 + k1][alpha] * V2[j2*2 + k2][beta]
                    if v_val != 0:
                        V[row_v][t] = v_val
            
            # Build W: W[k*4+i][t] = W2[k1*2+i1][alpha] * W2[k2*2+i2][beta]
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for i in range(4):
                    i1 = i // 2
                    i2 = i % 2
                    row_w = k * 4 + i
                    w_val = W2[k1*2 + i1][alpha] * W2[k2*2 + i2][beta]
                    if w_val != 0:
                        W[row_w][t] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
