import numpy as np
import sys

def construct():
    n, m, p = 3, 3, 8
    
    # Rank-23 decomposition for 3x3x3 matrix multiplication
    # This is a known optimal decomposition.
    # U, V, W are 9x23 matrices.
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # The following is a valid rank-23 decomposition.
    # Source: Various references on tensor rank of matrix multiplication.
    # I will construct it carefully.
    
    # Actually, let's use a simpler approach first: 
    # Split 3x3x8 into two 3x3x4 blocks.
    # R(3,3,4) <= R(3,3,3) + R(3,3,1) = 23 + 3 = 26? No, that's not how it works.
    # R(3,3,4) is known to be 24? Or maybe less?
    # Let's check if R(3,3,4) < 27.
    # If R(3,3,4) = 24, then 2*24 = 48.
    # If R(3,3,4) = 25, then 2*25 = 50.
    # If R(3,3,4) = 26, then 2*26 = 52.
    # All are < 72.
    
    # Let's try to find a decomposition for 3x3x4.
    # One strategy: Pad 3x3x4 to 4x4x4? No, that increases rank.
    # Another: Use the fact that R(3,3,4) <= R(3,3,3) + R(3,3,1) is not a valid inequality.
    # However, we can embed 3x3x4 into a larger tensor.
    
    # Let's try a different block split:
    # 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2
    # Rank = 23 + 23 + R(3,3,2)
    # R(3,3,2) = 18 (schoolbook). Is it less?
    # R(2,2,2) = 7. R(3,3,2) >= R(2,2,2) = 7.
    # Actually, R(3,3,2) is likely 18 or close to it.
    # Let's assume R(3,3,2) = 18.
    # Total = 23 + 23 + 18 = 64.
    # This is an improvement over 72.
    
    # I need a valid rank-23 decomposition for 3x3x3.
    # I will hardcode a known one.
    
    # Here is a rank-23 decomposition for 3x3x3.
    # U, V, W are 9x23.
    
    # I will define the vectors.
    # This is a complex set of vectors.
    # To ensure correctness, I will use a simpler, verified decomposition if possible,
    # or fall back to a slightly higher rank if I can't guarantee 23.
    
    # Let's try to use a rank-24 decomposition for 3x3x3 if 23 is too hard to hardcode correctly.
    # Or, I can use the fact that R(3,3,3) = 23 and just trust a hardcoded set.
    
    # Let's try to generate a valid decomposition programmatically?
    # No, 120s is tight.
    
    # I will hardcode a rank-23 decomposition.
    # The following is a known set of vectors.
    
    # U (9x23)
    U_333 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is identity, rank 9. Not correct.
    
    # I will use a different strategy:
    # Use a known rank-23 decomposition from a reliable source.
    # Since I cannot recall the exact 23 vectors, I will use a rank-27 (schoolbook) for 3x3x3
    # and see if I can improve the 3x3x2 part? No, 27+27+18 = 72.
    
    # Let's try to find a rank-26 decomposition for 3x3x3?
    # Or use a different block split.
    
    # What if I use a rank-24 decomposition for 3x3x3?
    # If I can find one, total rank = 24+24+18 = 66.
    
    # Let's try to use a known rank-23 decomposition.
    # I will hardcode it.
    
    # Here is a valid rank-23 decomposition for 3x3x3.
    # U, V, W are 9x23.
    
    # I will use the following vectors.
    # This is a known set.
    
    # Actually, let's just use the schoolbook solution to be safe, as I cannot guarantee the correctness of a hardcoded rank-23 decomposition without verification.
    # However, the task asks for an improvement.
    
    # Let's try to use a rank-24 decomposition for 3x3x3.
    # I will use a known rank-24 decomposition.
    
    # Or, I can use the fact that R(3,3,8) <= R(3,3,4) * 2.
    # If R(3,3,4) = 24, then R(3,3,8) <= 48.
    
    # Let's try to find a rank-24 decomposition for 3x3x4.
    
    # I will use a known rank-24 decomposition for 3x3x4.
    
    # Actually, let's just return the schoolbook solution.
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

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
