"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen-like decomposition for 4x4 x 4x9
    # Split into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B1, B2], [B3, B4]] where each is 2x9
    # C = [[C1, C2], [C3, C4]] where each is 2x9
    # C1 = A11*B1 + A12*B3, C2 = A11*B2 + A12*B4
    # C3 = A21*B1 + A22*B3, C4 = A21*B2 + A22*B4
    # Use Strassen for 2x2 x 2x9: 7 multiplications per 2x2 block pair
    # Total: 7 * 2 = 14 multiplications for the 4x4 x 4x9 case? No, that's not right.
    
    # Actually, for 4x4 x 4x9, we can view it as:
    # A is 4x4, B is 4x9
    # We can use the fact that 4x4 multiplication has rank at most 49
    # But for 4x4 x 4x9, we need a different approach
    
    # Let's use a direct construction with rank reduction
    # The key insight: for 4x4 x 4x9, we can achieve rank 7*4 + 4*9 - 4 = 44? No.
    
    # Better approach: use the known result that 4x4 x 4x4 has rank at most 49
    # For 4x4 x 4x9, we can use a block decomposition
    
    # Simplest: use schoolbook but with sharing
    # Actually, let's just use a known low-rank decomposition
    
    # For now, use a construction that achieves rank 7*9 + 4*4 = 77? 
    # Or use the fact that we can do 4x4 in rank 7 per 2x2 block
    
    # Let me implement a proper Strassen-based decomposition
    n, m, p = 4, 4, 9
    
    # For 4x4 x 4x9, we can split B into three 4x3 matrices
    # And use 4x4 x 4x3 with rank 7*3 = 21? No, that's not how it works.
    
    # Actually, the cleanest approach: use the fact that matrix multiplication
    # (4,4,4) has rank at most 49, and extend to (4,4,9)
    # Rank(4,4,9) <= 9/4 * 49 = 110.25, so at most 111? That's not helpful.
    
    # Let me just use a known good decomposition
    # For 4x4 x 4x9, a good upper bound is 7*4 + 4*5 = 48? 
    
    # I'll implement a construction based on tensor decomposition
    # Using the fact that we can share multiplications across the 9 columns
    
    rank = 7 * 4 + 4 * 5  # 28 + 20 = 48
    # Actually, let me use a simpler but valid construction
    
    # Use schoolbook: rank = 144
    # But we can do better with Strassen for the 2x2 blocks
    
    # For 4x4 x 4x9:
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x9 blocks: B1, B2, B3, B4 (each 2x9)
    # C1 = A11*B1 + A12*B3 (2x9)
    # C2 = A11*B2 + A12*B4 (2x9)  
    # C3 = A21*B1 + A22*B3 (2x9)
    # C4 = A21*B2 + A22*B4 (2x9)
    
    # Each 2x2 x 2x9 multiplication can be done with 7 multiplications (Strassen)
    # So total rank = 4 * 7 = 28
    
    U = []
    V = []
    W = []
    
    # For each of the 4 block multiplications, we need 7 rank-1 terms
    # Block 1: A11*B1 -> contributes to C1
    # Block 2: A12*B3 -> contributes to C1
    # Block 3: A11*B2 -> contributes to C2
    # Block 4: A12*B4 -> contributes to C2
    # Block 5: A21*B1 -> contributes to C3
    # Block 6: A22*B3 -> contributes to C3
    # Block 7: A21*B2 -> contributes to C4
    # Block 8: A22*B4 -> contributes to C4
    
    # Strassen for 2x2 x 2x9:
    # M1 = (A11+A22)*(B1+B3)  -> but B1, B3 are 2x9, so this is 2x9
    # Actually, Strassen works for 2x2 x 2x2. For 2x2 x 2x9, we can apply Strassen
    # to each column of B, giving 7*9 = 63 multiplications. That's worse than schoolbook (36).
    
    # So Strassen doesn't help for 2x2 x 2x9 directly.
    
    # Let me reconsider. The schoolbook for 4x4 x 4x9 is 144.
    # We can't easily reduce this without a clever decomposition.
    
    # For now, return the schoolbook decomposition (rank 144)
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
