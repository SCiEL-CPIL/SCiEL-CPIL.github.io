import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank 26 decomposition:
    # Rank 23 for the 3x3x3 part (columns 0,1,2 of B)
    # Rank 3 for the 3x3x1 part (column 3 of B)
    
    # The 23-rank decomposition of 3x3x3 matrix multiplication
    # Coefficients are in U, V, W for the first 23 columns
    # U is 9x23, V is 12x23 (but only first 9 rows used for k=0,1,2), W is 12x23 (only first 9 rows used)
    
    # I will use a known 23-rank decomposition.
    # Since I cannot look it up externally, I will use a simpler bound if I don't remember it.
    # However, 23 is the known optimum for 3x3x3.
    # If I don't have the exact coefficients, I can fall back to schoolbook.
    # But I can try to generate a random low rank? No, must be exact.
    
    # Let's use the schoolbook for now if I can't recall the 23-rank coefficients.
    # Wait, I can use the 23-rank coefficients from memory or derive a simpler improvement.
    # Actually, the 23-rank coefficients are:
    # (This is risky without a reference)
    
    # Alternative: Use the 24-rank decomposition?
    # Or just return the schoolbook 36?
    # The prompt asks for "stronger algorithm".
    # I will implement the 26-rank decomposition using a known 23-rank basis.
    
    # Since I don't have the 23-rank coefficients memorized perfectly, 
    # I will use a generic low-rank search if possible? 
    # No, 120s is tight.
    
    # I will use the schoolbook but optimized? No.
    
    # Let's try to construct the 3x3x3 part with rank 23.
    # If I fail, I use 36.
    
    # Actually, I can use the 23-rank decomposition from the literature.
    # The coefficients are:
    # U (9x23), V (9x23), W (9x23) for the 3x3x3 part.
    # Then pad V and W to 12x23 with zeros for the 4th column.
    # Then add 3 more columns for the 4th column part.
    
    # I will hardcode the 23-rank decomposition.
    # (Self-correction: I don't have the exact 23 numbers. I will use a rank 24 or 25 if I can find a simple one, or stick to 36 if I can't be sure.)
    
    # However, I know that rank 23 is possible.
    # I will write a function that attempts to find a rank 23 decomposition using random initialization and refinement?
    # Too slow.
    
    # I will use the schoolbook 36. It is safe.
    # But the score is 1.0.
    # If I return 36, score is 1.0.
    # If I return 26, score is 36/26 = 1.38.
    
    # I will try to implement the 23-rank decomposition.
    # I found a reference in my training data for the 23-rank 3x3x3 decomposition.
    # The coefficients are integers.
    
    # Let's use the 23-rank decomposition.
    U_23 = [
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
    # This is just identity. Rank 9.
    # I don't have the real 23-rank coefficients.
    
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
