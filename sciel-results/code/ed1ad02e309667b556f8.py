"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,3,7} into M_{2,3,3} and M_{2,3,4}
    # Rank of M_{2,3,3} is 12
    # Rank of M_{2,3,4} is 20 (two M_{2,2,3} of rank 10)
    # Total rank 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w

    # M_{2,3,3} part (cols 0-2 of V, rows 0-2 of W)
    # Using rank 12 algorithm for 2x3x3
    # This is a known decomposition.
    # I will use a standard rank 12 set.
    # For safety, I will use schoolbook for the 2x3x3 part (rank 18) 
    # and schoolbook for 2x3x4 (rank 24) -> 42. 
    # This is not an improvement.
    
    # I must use the fast algorithms.
    # I will use the rank 10 algorithm for 2x2x3.
    # And rank 12 for 2x3x3.
    
    # Since I cannot guarantee the exact coefficients for rank 10/12 without lookup,
    # I will use a rank 34 decomposition which is easier to verify?
    # No, I will use rank 32 with hardcoded coefficients from a reliable source.
    
    # Coefficients for M_{2,2,3} rank 10 (Winograd)
    # u, v are 2D, w is 3D (for the 3 columns of B)
    # Actually, the tensor is U (2), V (2), W (3).
    # Indices: U[0..1], V[0..1], W[0..2]
    # The 10 terms are:
    # 1. (1,0), (1,0), (1,0,0)
    # 2. (0,1), (1,0), (0,1,0)
    # 3. (1,0), (0,1), (0,0,1)
    # 4. (0,1), (0,1), (1,0,0)
    # 5. (1,1), (1,1), (1,1,1)
    # 6. (1,-1), (1,1), (1,-1,0)
    # 7. (1,1), (1,-1), (0,1,-1)
    # 8. (1,0), (1,1), (0,0,1)
    # 9. (0,1), (1,1), (1,0,0)
    # 10. (1,1), (1,0), (0,1,0)
    # This is a guess. I will not use it.
    
    # I will use the schoolbook decomposition for the whole thing but with a twist?
    # No.
    
    # I will return the schoolbook decomposition.
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
