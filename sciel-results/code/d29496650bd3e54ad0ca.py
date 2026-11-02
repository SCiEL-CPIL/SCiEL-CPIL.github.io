"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 2x2x5 matrix multiplication, the bilinear rank (tensor rank) is 2*p = 10? 
    # No, the schoolbook decomposition has rank n*m*p = 2*2*5 = 20.
    # The minimal rank for 2x2xn is 2n for n>=1, so for n=5, rank=10? 
    # Wait, that doesn't sound right. Let me reconsider.
    # For 1x1x1, rank=1. For 2x2x1, rank=4 (the 4 scalar mults).
    # For 2x2xn, each column is an independent 2x2x1 problem, so rank = 4*n? 
    # But that's 20 for n=5. Can we share?
    # Actually, I believe the rank of 2x2xn matrix multiplication is 2n for the bilinear complexity.
    # Let me verify: for 2x2x1, rank should be 4, not 2. So 2n doesn't work.
    # The schoolbook rank is n*m*p. For 2x2x5, that's 20.
    # I'll use the schoolbook decomposition with rank 20.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = (i * m + j) * p + k
                U[i * m + j][s] = 1
                V[j * p + k][s] = 1
                W[k * n + i][s] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
