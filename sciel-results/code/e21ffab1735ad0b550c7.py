"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll use Strassen-style decomposition for the 4x4 part.
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x9) into 2x2 blocks: B11, B12, B21, B22 where each is 2x9
    # 
    # Using 7 Strassen products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7  (2x9)
    # C12 = M3 + M5             (2x9)
    # C21 = M2 + M4             (2x9)
    # C22 = M1 - M2 + M3 + M6   (2x9)
    #
    # Each Mi is a 2x9 result. We need to decompose each 2x9 product.
    # For 2x2 * 2x9, we can use Strassen with 7 terms, or schoolbook with 4 terms.
    # Let's use schoolbook for the inner products: 2*2 = 4 terms per Mi
    # Total: 7 * 4 = 28 terms? No, that's not right.
    
    # Actually, let me think more carefully.
    # A is 4x4, B is 4x9.
    # If I split A into 2x2 blocks and B into 2x2 blocks (each 2x9),
    # then each Mi = (2x2 block of A) * (2x9 block of B) gives a 2x9 result.
    # To compute a 2x2 * 2x9 product, I can use 4 scalar multiplications per output column pair.
    # Actually, 2x2 * 2x9: for each of the 9 columns, I do a 2x2 * 2x1 = 4 mults.
    # So 9 * 4 = 36 mults per Mi, times 7 Mi = 252. That's worse than 144.
    
    # Let me try a different approach. Use Strassen at the top level only.
    # The 7 products Mi each involve 2x2 matrices times 2x9 matrices.
    # For each Mi, I need rank to compute 2x2 * 2x9.
    # Rank of 2x2*2x9 matrix mult: at most 2*2*9 = 36, but can be less.
    # Actually, for 2x2 * 2xp, the rank is at most 4p (schoolbook).
    # But we can do better.
    
    # Let me just use schoolbook for everything but try to be smarter.
    # Actually, let me try a direct Strassen decomposition.
    
    # For 4x4 * 4x9:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bj is 2x3
    # 
    # Wait, B is 4x9, so split into 2x2 blocks vertically:
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bjk is 2x3
    # 
    # Then C = A*B is 4x9:
    # C = [[C11, C12, C13], [C21, C22, C23]] where each Cjk is 2x3
    # 
    # Using Strassen:
    # M1 = (A11+A22)*(B11+B21+B21+B21) -- this doesn't work cleanly because B has 3 column blocks.
    
    # Let me pad B to 4x12 by adding 3 zero columns, then use 4x4 Strassen, then discard.
    # That gives rank 49 for the 4x4 part, but we're computing extra.
    # Actually, 4x4 * 4x12 with Strassen: 49 * (rank of 2x2*2x6)
    # This is getting complicated.
    
    # Let me try a simpler approach: just use schoolbook but verify it's correct.
    # Schoolbook: rank = 4*4*9 = 144
    
    # Actually, let me try to use a known lower rank.
    # For 4x4 * 4x4, the best known is 49 (Strassen).
    # For 4x4 * 4x9, I can think of it as 4x4 * 4x4 * (4x4 to 4x9 projection)
    # but that doesn't directly help.
    
    # Let me try: decompose 4x4*4x9 as (4x4*4x4) composed with a linear map.
    # If I have a rank-r decomposition of 4x4*4x4, I can extend it.
    
    # For now, let me implement Strassen for 4x4*4x4 (rank 49) and then extend to 4x9.
    # Actually, I'll just use schoolbook for 4x4*4x9 which is 144.
    
    # But wait - can I do better than 144?
    # Let me try: use Strassen on the 4x4 part, treating B as 4x4 with 3 extra columns.
    
    # Alternative: use the fact that 4x4*4x9 can be seen as 3 copies of 4x4*4x3 plus adjustments.
    # This is getting too complex. Let me just return schoolbook.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col in range(rank):
        k = col % p
        tmp = col // p
        j = tmp % m
        i = tmp // m
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
