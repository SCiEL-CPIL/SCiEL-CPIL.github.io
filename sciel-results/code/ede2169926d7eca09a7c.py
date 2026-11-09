"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a hybrid decomposition.
    # We split the indices into a 2x2x2 block and the rest.
    # The 2x2x2 block uses Laderman's algorithm (rank 7).
    # The rest uses schoolbook (rank 97).
    # Total rank = 104.
    
    rank = 104
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    
    # Laderman's algorithm for 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = a11*(b11+b22), M2 = a22*(b11+b22), M3 = a11*(a12-a22) ... wait, Laderman is for matrix mult.
    # C = A * B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a12+a22)(b21+b22)
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)(b21-b11) -- wait, this is not standard Laderman.
    
    # Standard Laderman:
    # m1 = a11 * (b11 + b22)
    # m2 = a22 * (b11 + b22)
    # m3 = a11 * (a12 - a22) -- No, this is not right.
    
    # Let's use the known 7-term decomposition for 2x2x2.
    # U, V, W for the 2x2x2 block.
    # We map the local indices (i,j,k) in {0,1}^3 to the global indices.
    # i_global = i_local, j_global = j_local, k_global = k_local.
    
    # Laderman's terms:
    # 1. U: a11, V: b11+b22, W: c11+c22? No.
    
    # Let's just use a known 7-term decomposition.
    # Terms:
    # 1. U: [1, 0, 0, 0] (a11), V: [1, 0, 1, 0] (b11+b22), W: [1, 0, 1, 0] (c11+c22) -- No, W is output.
    
    # The tensor is T[i,j,k] = delta[i,j,k] for the identity? No, T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the indices match? No.
    # The tensor is defined by T[i,j,k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # So for the 2x2x2 block, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need to decompose the 8-ones tensor into 7 rank-1 tensors.
    
    # Laderman's decomposition for the 2x2x2 tensor of all 1s:
    # 1. U=[1,0,0,0], V=[1,0,1,0], W=[1,0,0,0] -> a11(b11+b22) -> contributes to c11, c12?
    # This is getting confusing. Let's use a simpler approach.
    
    # Since I don't have the exact Laderman coefficients memorized for the tensor form, 
    # I will stick to the schoolbook decomposition to ensure correctness, 
    # but I will try to find a linear dependency to reduce the rank by 1.
    
    # For now, I will return the schoolbook decomposition.
    # This is safe and correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
