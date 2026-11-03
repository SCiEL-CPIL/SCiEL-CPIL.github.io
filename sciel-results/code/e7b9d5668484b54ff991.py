"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Rank-66 construction using Strassen-style blocks
    # Split m=4 into 2+2, p=6 into 3+3
    # This gives 4 sub-blocks of (3x2) x (2x3)
    # Each sub-block has schoolbook rank 18, total 72
    # Use Strassen on pairs to reduce
    
    # Actually, let me use a direct rank-66 construction
    # Based on known results for small matrix multiplication
    
    # Try: use the fact that we can share some computations
    # Rank 66 = 72 - 6, saving 6 multiplications
    
    # Construct using block Strassen:
    # Split A (3x4) into A1 (3x2) | A2 (3x2)
    # Split B (4x6) into B1 (2x3) ; B2 (2x3) ; B3 (2x3)
    # Wait, B is 4x6, split rows 2+2, cols 3+3:
    # B = [B11 B12; B21 B22] where each Bi is 2x3
    
    # C = A*B = A1*B1 + A2*B2 where B1 = [B11 B12], B2 = [B21 B22]
    # So C = A1*[B11 B12] + A2*[B21 B22]
    # = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # Each term is (3x2)*(2x3) = 3x3, schoolbook rank 6
    # 4 such terms: rank 24? No wait...
    
    # Actually let me just do schoolbook but with some sharing
    # For now, return the known rank-66 construction
    
    # I'll construct a valid rank-66 decomposition
    rank = 66
    U = [[0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * rank for _ in range(p * n)]  # 18 x 66
    
    # Schoolbook gives rank 72. We need to save 6.
    # Use the fact that for (3x2)*(2x3), we can use rank 5 instead of 6?
    # No, that's not generally possible.
    
    # Let me just return schoolbook for now and verify correctness
    rank = n * m * p  # 72
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
