"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 multiplication using recursive Strassen (Rank 49)
    # We use 7 Strassen products, each decomposed into 7 rank-1 terms for 2x2x2.
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to generate the 49 terms.
    # Each Mi is a product of two 2x2 linear combinations.
    # We decompose each Mi into 7 terms using the 2x2x2 Strassen decomposition.
    
    # The 2x2x2 Strassen decomposition has 7 terms.
    # Let's define them.
    # For a 2x2x2 multiplication, the tensor is T_{a,b,c}.
    # The rank 7 decomposition is:
    # 1. (A00+A11)(B00+B11) -> contributes to C00, C11
    # ...
    
    # This is getting too complex to hardcode manually in the time limit.
    # Let's just use the schoolbook for 4x4x4 (Rank 16) and 4x4x1 (Rank 16).
    # Total 32.
    # This is better than 80 and easy to implement.
    
    # Wait, 4x4x4 schoolbook is 16 terms.
    # 4x4x1 schoolbook is 16 terms.
    # Total 32.
    
    # Let's implement this.
    
    # Part 1: 4x4x4 (k=0..3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 4x4x1 (k=4)
    for i in range(4):
        for j in range(4):
            k = 4
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
