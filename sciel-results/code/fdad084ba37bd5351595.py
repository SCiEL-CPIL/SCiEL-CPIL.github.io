"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    # Schoolbook rank: n*m*p = 60
    # We try to find a better decomposition using a structured approach.
    # For 2x3 * 3x10, we can use a decomposition based on the fact that
    # the middle dimension is 3, which allows for some savings.
    
    # A known rank-47 decomposition exists for 2x2 * 2x2 (Strassen),
    # but here we have 2x3 * 3x10.
    # Let's use a heuristic search to find a rank < 60 decomposition.
    
    # Actually, let's use a simple constructive approach.
    # We can decompose the 3x3 multiplication in the middle using Strassen-like methods.
    # But the dimensions are 2x3 and 3x10, so it's not directly applicable.
    
    # Let's try a different approach: use a rank-1 decomposition for each output element,
    # but share computations where possible.
    
    # For now, let's return the schoolbook decomposition to ensure correctness.
    # We can improve this in future iterations.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
