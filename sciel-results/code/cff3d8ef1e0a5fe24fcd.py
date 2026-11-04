"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # Decomposition:
    # Split m=3 into m1=2 and m2=1.
    # Part 1: A1 (2x2) * B1 (2x5). Rank 10.
    # Part 2: A2 (2x1) * B2 (1x5). Rank 1.
    # Total Rank: 11.
    
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (rows 0,1 of A; cols 0,1) * B1 (rows 0,1 of B; cols 0..4)
    # Indices: i in 0..1, j in 0..1, k in 0..4
    # We need 2 terms for each (i,k) pair.
    # Term 1: j=0. U[i,0]=1, V[0,k]=1, W[k,i]=1
    # Term 2: j=1. U[i,1]=1, V[1,k]=1, W[k,i]=1
    
    for i in range(n): # i in 0,1
        for k in range(p): # k in 0..4
            # Term for j=0
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
            # Term for j=1
            U[i * m + 1][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 (rows 0,1 of A; col 2) * B2 (row 2 of B; cols 0..4)
    # Indices: i in 0..1, j=2, k in 0..4
    # This is rank 1.
    # U[i,2]=1, V[2,k]=1, W[k,i]=1
    
    for i in range(n):
        U[i * m + 2][col] = 1
        
    for k in range(p):
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1 # Wait, W depends on i too.
        
    # Correction for Part 2:
    # The term is U[i,2] V[2,k] W[k,i].
    # We need W[k,i] to be 1 for all i,k?
    # If we set W[k,i]=1 for all i,k, then for a fixed i, the sum is 1.
    # But U[i,2] is 1 for all i.
    # So for any i,k, U[i,2]=1, V[2,k]=1, W[k,i]=1 -> Product 1.
    # So we just need to fill W[k * n + i] with 1 for all i,k.
    
    # But I already incremented col? No, I only used one column index 'col' for the whole part.
    # Let's redo Part 2 carefully.
    # We have 1 column left (col is 10).
    # U[i * m + 2][col] = 1 for i in 0,1.
    # V[2 * p + k][col] = 1 for k in 0..4.
    # W[k * n + i][col] = 1 for i in 0,1 and k in 0..4.
    
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
