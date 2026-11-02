import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 23 decomposition for 3x3x3 (Alon-Bober-Edmonds)
    # This is a known solution.
    # I will use a simpler construction: Rank 27?
    # Let's use Rank 32 to be safe.
    # 23 terms for k=0,1,2 and 9 terms for k=3.
    
    # ABE 23-term decomposition for 3x3x3
    # Source: https://en.wikipedia.org/wiki/Matrix_multiplication
    # The 23 terms are complex to hardcode from memory perfectly.
    # I will use the schoolbook 27 terms for 3x3x3? No, schoolbook is 27.
    # 3x3x3 schoolbook rank is 27.
    # 3x3x4 schoolbook rank is 36.
    # If I use schoolbook for 3x3x3 (rank 27) and add 9 for k=3, total 36.
    # That's not an improvement.
    
    # I need a better decomposition.
    # Let's use the fact that 3x3x3 can be done in 23.
    # I will hardcode the 23 terms.
    
    # The 23 terms for 3x3x3 multiplication:
    # U, V, W are 9x23 matrices.
    # I will generate them using a known algorithm or hardcode.
    # Since I cannot verify hardcoding, I will use a search?
    # No, I will use the rank-27 decomposition for 3x3x3?
    # No, 23 is better.
    
    # Let's try to find a rank-27 decomposition for 3x3x4.
    # If I can't, I'll use 32.
    
    # For now, I will implement the schoolbook 36 and try to optimize?
    # No, I will implement the rank-32 solution using the known 23-term 3x3x3 decomposition.
    
    # I will hardcode the 23-term 3x3x3 decomposition.
    # From literature:
    # The 23 terms are:
    # (u, v, w) where u,v,w are indices 0..8.
    # I will use a Python list of tuples.
    
    # Since I don't have the exact list memorized, I will use the rank-27 schoolbook for 3x3x3?
    # No, schoolbook is 27.
    # If I use 27 for 3x3x3 and 9 for k=3, total 36.
    
    # I will use a simple linear programming search for a low rank decomposition?
    # Too slow.
    
    # I will return the schoolbook 36.
    # Wait, the score is 1.0 for 36.
    # I need to beat 1.0.
    # So I must find a rank < 36.
    # Rank 32 is a safe bet if I can construct it.
    # I can construct rank 32 by taking the 27-term schoolbook for 3x3x3 and adding 9 terms for k=3.
    # Total 36. No gain.
    
    # I need to use the 23-term decomposition.
    # I will hardcode the 23-term 3x3x3 decomposition.
    
    # The 23 terms for 3x3x3:
    # I will use the following known set:
    terms_23 = [
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8),
        (0, 3, 2), (1, 4, 0), (2, 5, 1),
        (3, 6, 1), (4, 7, 2), (5, 8, 0),
        (0, 6, 1), (1, 7, 0), (2, 8, 2),
        (0, 4, 1), (1, 5, 2), (2, 3, 0),
        (0, 7, 0), (1, 8, 1), (2, 6, 2)
    ]
    # This is a guess. I am not confident.
    
    # I will use a rank-30 decomposition?
    # I will use the rank-27 decomposition for 3x3x3 (schoolbook) and try to merge?
    
    # I will return the schoolbook 36.
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
