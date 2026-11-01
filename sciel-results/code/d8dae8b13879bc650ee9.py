import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen forms for 2x2 multiplication
    # u_t, v_t are 2x2 matrices of coefficients for A and B blocks
    # w_t is 2x2 matrix of coefficients for C blocks
    
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # t=1: (A11+A22)(B11+B22) -> C11+=1, C22+=1
    # t=2: (A21+A22)B11 -> C21+=1, C22-=1
    # t=3: A11(B12-B22) -> C12+=1, C22+=1
    # t=4: A22(B21-B11) -> C11+=1, C21+=1
    # t=5: (A11+A12)B22 -> C11-=1, C12+=1
    # t=6: (A21-A11)(B11+B12) -> C22+=1
    # t=7: (A12-A22)(B21+B22) -> C11+=1
    
    # Representing u_t, v_t, w_t as 2x2 matrices
    # u_t[a,b] is coeff of A_ab
    # v_t[a,b] is coeff of B_ab
    # w_t[a,b] is coeff of C_ab
    
    u_forms = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]],# 6
        [[0, 1], [0, -1]] # 7
    ]
    
    v_forms = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 1], [0, -1]],# 3
        [[-1, 0], [0, 1]],# 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 0], [1, 1]]  # 7
    ]
    
    w_forms = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, -1]],# 2
        [[0, 1], [0, 1]], # 3
        [[1, 0], [1, 0]], # 4
        [[-1, 1], [0, 0]],# 5
        [[0, 0], [0, 1]], # 6
        [[1, 0], [0, 0]]  # 7
    ]
    
    # Outer Strassen forms for 4x4 (blocks are 2x2)
    # X_s, Y_s are 2x2 matrices of blocks
    # X_s = sum alpha_s_ab A_ab
    # Y_s = sum beta_s_cd B_cd
    
    # s=1: (A11+A22)(B11+B22)
    # s=2: (A21+A22)B11
    # s=3: A11(B12-B22)
    # s=4: A22(B21-B11)
    # s=5: (A11+A12)B22
    # s=6: (A21-A11)(B11+B12)
    # s=7: (A12-A22)(B21+B22)
    
    # alpha_s_ab is coeff of A_ab in X_s
    # beta_s_cd is coeff of B_cd in Y_s
    
    x_forms = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]],# 6
        [[0, 1], [0, -1]] # 7
    ]
    
    y_forms = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 1], [0, -1]],# 3
        [[-1, 0], [0, 1]],# 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 0], [1, 1]]  # 7
    ]
    
    # Outer coefficients: C_block = sum_s gamma_s_bc M_s
    # C = [[C11, C12], [C21, C22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    gamma = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, -1]],# 2
        [[0, 1], [0, 1]], # 3
        [[1, 0], [1, 0]], # 4
        [[-1, 1], [0, 0]],# 5
        [[0, 0], [0, 1]], # 6
        [[1, 0], [0, 0]]  # 7
    ]
    
    # Construct U, V, W
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U: coeff of A_ij
            # U_ij = sum_ab u_t_ab * x_s_ab? 
            # Wait, X_s is a matrix. u_t is a linear form on X_s.
            # u_t(X_s) = sum_ab u_t_ab * (X_s)_ab
            # (X_s)_ab = sum_ij x_s_ab_ij A_ij? 
            # No, x_s_ab is the coefficient of block A_ab in X_s.
            # X_s_ab = sum_{i in block a, j in block b} A_ij * coeff?
            # No, X_s is a 2x2 matrix of 2x2 blocks.
            # X_s_ab is a 2x2 matrix.
            # X_s_ab = sum_{i=0..1, j=0..1} A_{2a+i, 2b+j} * x_s_ab?
            # No, x_s_ab is a scalar coefficient for the whole block A_ab.
            # So X_s_ab = x_s_ab * A_ab.
            # Then u_t(X_s) = sum_ab u_t_ab * x_s_ab * A_ab.
            # So U_ij = sum_ab u_t_ab * x_s_ab * delta(i in block a, j in block b).
            
            for i in range(4):
                for j in range(4):
                    a = i // 2
                    b = j // 2
                    # Check if A_ij is in block A_ab
                    # Yes, if i=2a or 2a+1, j=2b or 2b+1
                    # The coefficient is u_t_ab * x_s_ab
                    U[i * 4 + j][r] = u_forms[t][a][b] * x_forms[s][a][b]
                    
            for j in range(4):
                for k in range(4):
                    b = j // 2
                    c = k // 2
                    V[j * 4 + k][r] = v_forms[t][b][c] * y_forms[s][b][c]
                    
            for i in range(4):
                for k in range(4):
                    # W index: k * 4 + i
                    # C_ik = sum_s gamma_s_bc * (M_s)_bc
                    # (M_s)_bc = sum_t u_t_ab v_t_bc w_t_bc? No.
                    # M_s = sum_t (u_t X_s) (v_t Y_s) w_t
                    # So (M_s)_bc = sum_t (u_t X_s) (v_t Y_s) w_t_bc
                    # C_ik = sum_s sum_t gamma_s_bc (u_t X_s) (v_t Y_s) w_t_bc
                    # where b = i//2, c = k//2
                    b = i // 2
                    c = k // 2
                    # Coeff is gamma_s_bc * w_t_bc
                    W[k * 4 + i][r] = gamma[s][b][c] * w_forms[t][b][c]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
