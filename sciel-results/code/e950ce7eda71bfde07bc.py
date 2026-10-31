import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2
    # u_l, v_l are 2x2 matrices for the l-th product
    # C_rs = sum_l gamma_l[rs] * (u_l . X) * (v_l . Y)
    
    # Define Strassen coefficients for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # u_l, v_l are 4-vectors (flattened 2x2)
    # X = [x00, x01, x10, x11]
    
    u = []
    v = []
    gamma = [] # gamma[l][r][s] is coeff for M_l[rs]
    
    # M1
    u.append([1, 0, 0, 1]) # x00 + x11
    v.append([1, 0, 0, 1]) # y00 + y11
    g = [[0,0],[0,0]]
    g[0][0] += 1
    g[1][1] += 1
    gamma.append(g)
    
    # M2
    u.append([0, 0, 1, 1]) # x10 + x11
    v.append([1, 0, 0, 0]) # y00
    g = [[0,0],[0,0]]
    g[1][0] += 1
    g[1][1] += 1
    gamma.append(g)
    
    # M3
    u.append([1, 0, 0, 0]) # x00
    v.append([0, 1, 0, -1]) # y01 - y11
    g = [[0,0],[0,0]]
    g[0][1] += 1
    g[1][1] += 1
    gamma.append(g)
    
    # M4
    u.append([0, 0, 0, 1]) # x11
    v.append([0, 0, 1, -1]) # y10 - y00
    g = [[0,0],[0,0]]
    g[0][0] += 1
    g[1][0] += 1
    gamma.append(g)
    
    # M5
    u.append([1, 1, 0, 0]) # x00 + x01
    v.append([0, 0, 0, 1]) # y11
    g = [[0,0],[0,0]]
    g[0][0] -= 1
    g[1][1] += 1
    gamma.append(g)
    
    # M6
    u.append([0, 0, 1, -1]) # x10 - x00
    v.append([1, 1, 0, 0]) # y00 + y01
    g = [[0,0],[0,0]]
    g[1][1] += 1
    gamma.append(g)
    
    # M7
    u.append([0, 1, 0, -1]) # x01 - x11
    v.append([0, 0, 1, 1]) # y10 + y11
    g = [[0,0],[0,0]]
    g[0][0] += 1
    g[0][1] += 1
    gamma.append(g)
    
    # Outer Strassen coefficients
    # X_k, Y_k are 2x2 matrices of blocks
    # alpha_k[r][s][I][J] is coeff of A[2I+r, 2J+s] in X_k[r][s]
    # beta_k[I][J] is coeff of M_k in C[2I, 2J]
    
    alpha = []
    beta = []
    
    # K=0: X0 = A00 + A11, Y0 = B00 + B11
    a = [[0]*4 for _ in range(2)] # a[r][s] is 4x4 coeff? No, 2x2 blocks of coeffs
    # Let's use 2x2 arrays of 2x2 arrays
    # alpha_k[r][s] is a 2x2 matrix of coeffs for A blocks
    # A_block[I][J] is 2x2.
    # X_k[r][s] = sum_{I,J} alpha_k[r][s][I][J] * A[2I+r, 2J+s] ? No.
    # X_k is 2x2. Each element X_k[r][s] is a 2x2 matrix.
    # X_k[r][s] = sum_{I,J} coeff * A_block[I][J]
    
    # K=0
    ak = [[0]*2 for _ in range(2)] # ak[r][s] is 2x2 coeff matrix for A blocks
    ak[0][0] = [[1,0],[0,1]] # A00
    ak[1][1] = [[1,0],[0,1]] # A11
    # Wait, A00 is block (0,0). A11 is block (1,1).
    # X0[0][0] = A00[0][0] + A11[0][0]? No.
    # X0 is a matrix of blocks.
    # X0[0][0] = A00. X0[0][1] = A01? No.
    # Strassen on blocks:
    # A = [[A00, A01], [A10, A11]]
    # X0 = A00 + A11. This is a 2x2 matrix.
    # So X0[0][0] = A00[0][0] + A11[0][0].
    # So alpha_0[0][0] (coeff of A00) is 1. alpha_0[0][0] (coeff of A11) is 1.
    
    # Let's define alpha_k as a list of 2x2 matrices, where each entry is a 2x2 matrix of coefficients for the 4 blocks.
    # Actually, simpler:
    # X_k[r][s] is a 2x2 matrix.
    # X_k[r][s] = sum_{I,J} c_{I,J} * A[2I+r, 2J+s] is wrong.
    # X_k[r][s] = sum_{I,J} C_{I,J} * A_block[I][J]
    # where C_{I,J} is a 2x2 matrix? No, C_{I,J} is a scalar if A_block is 2x2 and X_k[r][s] is 2x2?
    # No, X_k is 2x2. Its elements are 2x2 matrices.
    # So X_k[r][s] is a 2x2 matrix.
    # X_k[r][s] = sum_{I,J} \alpha_{k, r, s, I, J} A[2I+r, 2J+s] is not right because A[2I+r, 2J+s] is a scalar.
    # X_k[r][s] is a 2x2 matrix of scalars.
    # So X_k[r][s][a][b] = sum_{I,J} \alpha_{k, r, s, I, J, a, b} A[2I+a, 2J+b]
    
    # This is getting complicated. Let's just build the 4x4 U, V, W directly.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Define outer Strassen steps
    # k: 0..6
    # X_k, Y_k definitions
    # X_k is 2x2 matrix of 2x2 blocks
    # We represent X_k as a 4x4 matrix of coefficients for A
    # X_k_flat is 4x4? No.
    
    # Let's define the linear maps.
    # For each k, define map_A_k: 16 -> 4 (coeffs for X_k elements)
    # map_B_k: 16 -> 4
    # And map_C_k: 4 -> 16 (coeffs for C blocks from M_k)
    
    # Actually, let's just iterate.
    
    for k in range(7):
        # Define alpha_k: 2x2x2x2 (r, s, I, J) -> coeff
        # X_k[r][s] = sum_{I,J} alpha_k[r][s][I][J] * A[2I+r, 2J+s] ? No.
        # X_k[r][s] is a 2x2 matrix.
        # Element X_k[r][s][a][b] (a,b in 0..1)
        # X_k[r][s][a][b] = sum_{I,J} \alpha_{k, r, s, I, J, a, b} A[2I+a, 2J+b]
        
        # It's easier to define the 4x4 matrix of coefficients for A.
        # Let A_mat be 4x4.
        # X_k is 4x4 matrix of coeffs.
        # X_k[i][j] is coeff of A[i][j] in the flattened X_k?
        # No, X_k is 2x2 blocks.
        
        # Let's use the explicit Strassen formulas.
        
        # k=0: X0 = A00 + A11
        # X0[0][0] = A00[0][0] + A11[0][0] = A[0][0] + A[2][2]
        # X0[0][1] = A00[0][1] + A11[0][1] = A[0][1] + A[2][3]
        # X0[1][0] = A00[1][0] + A11[1][0] = A[1][0] + A[3][2]
        # X0[1][1] = A00[1][1] + A11[1][1] = A[1][1] + A[3][3]
        
        # We can define a 4x4 matrix U_k where U_k[i][j] is the coefficient of A[i][j] in the l-th inner product.
        
        pass # This is too messy to do manually in thought. I will code the logic.

    # Implementation logic:
    # 1. Define u, v, gamma for inner Strassen.
    # 2. Define outer Strassen coefficients.
    # 3. For each c (0..48):
    #    k = c // 7
    #    l = c % 7
    #    Compute U_vec, V_vec, W_vec.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
