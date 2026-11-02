"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix)
    # U indices: i * m + j
    U_coeffs = [
        [1, 0, 0, 1], # s=0: a+d
        [0, 0, 1, 1], # s=1: c+d
        [1, 0, 0, 0], # s=2: a
        [0, 0, 0, 1], # s=3: d
        [1, 1, 0, 0], # s=4: a+b
        [-1, 0, 1, 0],# s=5: c-a
        [0, 1, -1, 0] # s=6: b-d
    ]
    
    # Strassen coefficients for V (B matrix, local)
    # V indices: j * p + k
    # Local k=0,1 map to global 2b, 2b+1
    V_coeffs = [
        [1, 0, 0, 1], # s=0: e+h
        [1, 0, 0, 0], # s=1: e
        [0, -1, 1, 0],# s=2: f-g (Wait, M3 = a(f-h), so B part is f-h. f is B[0,1], h is B[1,1]. So V[0,1]=-1, V[1,1]=1. Correct.)
        [0, -1, 1, 0],# s=3: g-e (M4 = d(g-e). g is B[1,0], e is B[0,0]. So V[0,0]=-1, V[1,0]=1. Correct.)
        [0, 0, 0, 1], # s=4: h
        [1, 1, 0, 0], # s=5: e+f
        [0, 1, 1, 1]  # s=6: g+h
    ]
    
    # Strassen coefficients for W (C matrix, local)
    # W indices: k * n + i
    W_coeffs = [
        [1, 0, 0, 1], # s=0: C00, C11
        [0, 0, 1, -1],# s=1: C10, C11
        [0, 1, 0, 1], # s=2: C01, C11
        [0, 1, 0, 1], # s=3: C10, C11 (Wait, M4 contributes to C10 and C11. Correct.)
        [-1, 0, 1, 0],# s=4: C00, C01
        [0, 0, 0, 1], # s=5: C11
        [1, 0, 0, 0]  # s=6: C00
    ]
    
    for b in range(7):
        for s in range(7):
            r = 7 * b + s
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = U_coeffs[s][i * 2 + j]
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * p + k_global][r] = V_coeffs[s][j * 2 + k_local]
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * b + k_local
                    W[k_global * n + i][r] = W_coeffs[s][k_local * 2 + i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
