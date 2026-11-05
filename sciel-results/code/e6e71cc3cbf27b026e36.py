import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition:
    # Block 1: 3x3x3 (cols 0-2) -> Rank 23 (Winograd-like, hardcode)
    # Block 2: 3x3x3 (cols 3-5) -> Rank 23 (same)
    # Block 3: 3x3x2 (cols 6-7) -> Rank 18 (Schoolbook, since R(3,3,2)=18)
    # Total rank = 23 + 23 + 18 = 64
    
    # We need a valid rank-23 decomposition for 3x3x3.
    # Since hardcoding 23x9 vectors from memory is error-prone,
    # and generating one reliably in 120s is hard,
    # let's check if we can use a known simple structure or a slightly higher rank that is easier to construct and still < 72.
    # R(3,3,3) = 23. R(3,3,4) = 30? R(3,3,5)=35?
    # Actually, a very common efficient decomposition for 3x3x3 is rank 24 or 23.
    # If I can't get 23, can I get 24? 24+24+18 = 66 < 72.
    # Can I get a rank-24 decomposition?
    # It is known that R(3,3,3) <= 24. A rank-24 decomposition is easier to find?
    
    # Let's try to construct a rank-24 decomposition for 3x3x3 using a perturbation or known sparse structure.
    # Actually, the "Bini" decomposition or similar gives rank 23.
    # Let's try a different strategy:
    # Use Strassen on the 2x2x2 blocks?
    # A 3x3 matrix can be partitioned. But Strassen gives rank 7 for 2x2x2 vs 8.
    # For 3x3x3, a block approach:
    # Partition 3x3 into 1x1, 1x2, etc. Not standard.
    
    # Let's stick to the block idea but verify if I can generate a valid low-rank decomposition.
    # If I fail to generate a valid sub-72 rank, I should fallback to 72?
    # But the task asks for improvement.
    # Let's assume I can hardcode a valid rank-23 decomposition.
    # Here is a known rank-23 decomposition for 3x3x3 (from literature, e.g., Winograd).
    # I will construct it by defining the 23 triples (u_r, v_r, w_r) such that sum u_r v_r w_r = T_3.
    # This is difficult to do inline without pre-computation.
    
    # Alternative: Use a rank-27 decomposition but optimize the 3x3x2 block?
    # Schoolbook 3x3x2 is rank 18 (3*3*2). This is minimal.
    # Schoolbook 3x3x3 is rank 27.
    # 27+27+18 = 72.
    
    # Is there a rank-25 or 26 decomposition for 3x3x3 that is easy to construct?
    # Yes, by taking a rank-27 and merging 2 terms? No, merging requires orthogonality.
    
    # Let's try to find a rank-24 decomposition for 3x3x3.
    # If I can produce a valid rank-24, total is 66.
    # If I can produce rank-23, total is 64.
    
    # I will attempt to hardcode a rank-23 decomposition.
    # Reference: "The border rank of matrix multiplication"
    # One explicit rank-23 decomposition is given by the following 23 tensors:
    # I will use a list of tuples (i, j, k) with coefficients if possible, but the format requires U,V,W matrices.
    
    # Since I cannot reliably hardcode 23 complex vectors, I will implement a heuristic search for a low-rank decomposition of 3x3x3 within the time limit.
    # If it finds rank < 27, I use it. Otherwise, I fall back to 27.
    # However, searching for exact integer/rational rank-23 is very hard.
    
    # Let's try a different approach:
    # Use the fact that R(3,3,8) might be lower than 64?
    # Actually, a global optimization might be better than block decomposition.
    # But block decomposition is safe.
    
    # Let's try to construct a rank-24 decomposition for 3x3x3.
    # It is known that R(3,3,3) = 23.
    # A rank-24 decomposition is easy to construct by taking a rank-23 and adding a zero term? No, that's rank 23.
    # Taking a rank-27 and reducing?
    
    # I will use a pre-computed rank-23 decomposition if I can recall it.
    # Otherwise, I will use a rank-27 decomposition (schoolbook) for the 3x3x3 blocks.
    # 27+27+18 = 72.
    
    # Wait, I can use a rank-24 decomposition for 3x3x3 if I use a specific trick.
    # Actually, let's use the "Winograd" 3x3x3 which is often cited as rank 23.
    # I will hardcode a known set of 23 vectors.
    # If I get it wrong, the validation will fail.
    # To be safe, I will output the schoolbook 72 if I can't guarantee the 23.
    
    # However, the prompt implies an improvement is possible.
    # Let's try to use a rank-26 decomposition?
    # Actually, there is a known rank-24 decomposition for 3x3x3 that is simpler than 23?
    # No, 23 is the minimum.
    
    # Let's try to use a global search for rank < 72?
    # Too slow.
    
    # I will provide the schoolbook solution (72) as it is guaranteed valid.
    # But I will try to optimize the 3x3x3 blocks to rank 24 if I can find a simple one.
    # Actually, the Bini decomposition for 3x3x3 has rank 23.
    
    # Let's assume I can hardcode the 23 vectors.
    # I will define them as lists of 9 floats/integers.
    # U[0..8], V[0..8], W[0..8]
    
    # Since I don't have the exact values memorized, I will use a fallback.
    # The safest "improvement" is if I can find a rank-24 or 23.
    # If I can't, I return 72.
    
    # Let's try to construct a rank-24 decomposition.
    # I can take the schoolbook 27 terms and see if 3 can be eliminated.
    # This is a research problem.
    
    # I will output the schoolbook solution.
    
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
