"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Base 2x2 Strassen coefficients
    # alpha[k][i][j]
    alpha = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 0], [0, 1]],  # M3
        [[1, 1], [0, 0]],  # M4
        [[-1, 0], [1, 0]], # M5
        [[0, -1], [0, 1]]  # M6
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # M0
        [[1, 0], [0, 0]],  # M1
        [[0, 1], [0, -1]], # M2
        [[-1, 0], [1, 0]], # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, 1]]   # M6
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # M0 -> C00, C11
        [[0, 0], [1, -1]], # M1 -> C10, C11(-)
        [[1, 1], [0, 0]],  # M2 -> C01, C11
        [[1, 0], [1, 0]],  # M3 -> C00, C10
        [[-1, 1], [0, 0]], # M4 -> C00(-), C01
        [[0, 0], [0, 1]],  # M5 -> C11
        [[1, 0], [0, 0]]   # M6 -> C00
    ]
    
    # Correct gamma based on Strassen reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    gamma = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, -1]], # M1
        [[1, 1], [0, 0]],  # M2
        [[1, 0], [1, 0]],  # M3
        [[-1, 1], [0, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 0], [0, 0]]   # M6
    ]
    
    # Base 2x2 Strassen U, V, W for scalar products
    # For each k in 0..6, we have a "rank-1" term in the block sense.
    # But we need to decompose the scalar products inside?
    # No, the block Strassen IS the rank-7 decomposition for 2x2.
    # So we just use alpha, beta, gamma directly as the coefficients for the 2x2 case.
    
    # Construct global U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(7):
        for t_local in range(7):
            t = k * 7 + t_local
            
            # We are effectively using the t_local-th term of the 2x2 Strassen
            # for the k-th block product.
            
            # But wait, the 2x2 Strassen uses 7 terms.
            # The coefficients for these terms are alpha[k], beta[k], gamma[k]?
            # No, alpha[k] is the coefficient for the k-th block product.
            # The 2x2 Strassen decomposition of a single 2x2 product X*Y
            # uses 7 terms, let's call them s=0..6.
            # The coefficients for these terms are fixed (Strassen constants).
            
            # Let's define the base 2x2 Strassen constants.
            # These are the same as alpha, beta, gamma but for a single product.
            
            # Actually, the alpha, beta, gamma I defined above ARE the Strassen constants
            # for the 2x2 case.
            
            # So for the k-th block product, we use the same constants?
            # No, the k-th block product is A_k * B_k.
            # A_k is a linear combo of A blocks.
            # B_k is a linear combo of B blocks.
            
            # The decomposition of A_k * B_k uses the base Strassen constants.
            # Let's call the base constants A_s, B_s, C_s for s=0..6.
            
            # A_s = alpha[s]
            # B_s = beta[s]
            # C_s = gamma[s]
            
            # So for term t = k*7 + s:
            # U contributes from A_k and A_s.
            # A_k = sum_{R,C} alpha[k][R][C] A_block[R][C]
            # A_s (for the product) = sum_{u,v} alpha[s][u][v] X_u_v
            # where X = A_k.
            
            # So U_global[i*4+j, t] = alpha[k][i//2][j//2] * alpha[s][i%2][j%2]
            
            # Similarly for V and W.
            
            a_k = alpha[k]
            b_k = beta[k]
            c_k = gamma[k] # Not used for U, V
            
            a_s = alpha[t_local]
            b_s = beta[t_local]
            c_s = gamma[t_local]
            
            for i in range(n):
                for j in range(m):
                    R = i // 2
                    C = j // 2
                    u = i % 2
                    v = j % 2
                    U[i * m + j][t] = a_k[R][C] * a_s[u][v]
            
            for j in range(m):
                for k_idx in range(p):
                    R = j // 2
                    C = k_idx // 2
                    u = j % 2
                    v = k_idx % 2
                    V[j * p + k_idx][t] = b_k[R][C] * b_s[u][v]
            
            for k_idx in range(p):
                for i in range(n):
                    R = k_idx // 2
                    C = i // 2
                    u = k_idx % 2
                    v = i % 2
                    W[k_idx * n + i][t] = c_k[R][C] * c_s[u][v]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
