import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Attempt to use a known low-rank decomposition for 3x3x5 matrix multiplication.
    # The schoolbook rank is 45.
    # We will try to construct a decomposition using Strassen-like ideas or known algorithms.
    # However, finding a valid rank < 45 decomposition for 3x3x5 is non-trivial.
    # For 3x3x3, the best known is 23 (Landsberg).
    # For 3x3x5, we can pad the 5 dimension to 6? No, that increases rank.
    # We can use the fact that 3x3x5 is a sub-tensor of 3x3x6.
    # If we have a decomposition for 3x3x6 of rank R, we can restrict it to 3x3x5.
    # But we need to ensure the coefficients for the 5th column are zero? No, that's not how it works.
    
    # Let's stick to a verified algorithm.
    # The Winograd algorithm for 3x3 matrix multiplication has rank 23 (Landsberg's algorithm).
    # But we have 3x3x5.
    # We can compute C = A * B where A is 3x3 and B is 3x5.
    # We can split B into B1 (3x3) and B2 (3x2).
    # C = A*B1 + A*B2.
    # If we use a rank-23 algorithm for A*B1, and schoolbook (6) for A*B2, total rank = 23 + 6 = 29.
    # This is an improvement over 45.
    # Let's implement Landsberg's rank-23 algorithm for 3x3x3 and schoolbook for 3x3x2.
    
    # Landsberg's 3x3x3 algorithm (rank 23)
    # This is complex to hardcode. Let's use a simpler known improvement if possible.
    # Actually, the best known for 3x3x3 is 23.
    # Let's use the algorithm from "A Note on the Tensor Rank of 3x3x3" or similar.
    # Since I cannot reliably hardcode the 23-rank coefficients from memory without error,
    # and the directive asks for a specific approach, I will try to implement a Strassen-based
    # decomposition that might be easier to verify or is a known intermediate step.
    
    # Alternative: Use the Strassen algorithm for 2x2x2 blocks?
    # 3x3x5 can be viewed as overlapping 2x2x2 blocks.
    # This is also complex.
    
    # Let's try a different strategy:
    # We can use the fact that the matrix multiplication tensor for m x n x p
    # has a rank related to the Kronecker product of smaller tensors.
    # But 3 and 5 are prime.
    
    # Let's go back to the splitting idea: 3x3x5 = 3x3x3 + 3x3x2.
    # Rank(3x3x3) <= 23 (Landsberg).
    # Rank(3x3x2) = 6 (Schoolbook, since 3*3*2=18, but we can do better? No, 3x3x2 is likely 6 if we use Strassen on 2x2? No, 3x3x2 is just 3*2=6 multiplications if we fix the 3 dimension? No.
    # A is 3x3, B is 3x2. C is 3x2.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For fixed k, we are multiplying a 3x3 matrix by a 3x1 vector.
    # This is a 3x3x1 tensor. Rank is 3 (schoolbook).
    # So for 3x3x2, we do two such operations. Total rank 6.
    # So total rank = 23 + 6 = 29.
    
    # I need the explicit coefficients for Landsberg's 23-rank algorithm.
    # I will attempt to construct a valid 29-rank decomposition.
    # If I cannot get the 23-rank coefficients correct, I will fall back to a known 23-rank or lower.
    # Actually, there is a known 23-rank algorithm. Let's try to find a simple one.
    # If I fail, I will return the schoolbook 45-rank to ensure validity, but that scores 1.0.
    # The prompt says "improve".
    
    # Let's try to implement the 23-rank algorithm from the literature.
    # One such algorithm is given in "Lower bounds for bilinear complexity of matrix multiplication".
    # However, hardcoding 23 sets of coefficients for 3x3x3 is risky.
    
    # Let's try a different split: 3x3x5.
    # Maybe we can use a rank-15 algorithm for 3x3x3? No, lower bound is 19.
    
    # Let's assume the 23+6=29 strategy is correct and try to find the coefficients.
    # Since I don't have them memorized, I will use a placeholder that might be wrong,
    # or I will use a known simple improvement.
    
    # Wait, I can use the Strassen algorithm for 2x2x2 to handle parts of the 3x3x5.
    # But the dimensions don't align nicely.
    
    # Let's try to use the following:
    # The tensor rank of 3x3x3 is 23.
    # I will hardcode a known 23-rank decomposition if I can recall it.
    # If not, I will use the schoolbook for 3x3x3 (rank 27) + schoolbook for 3x3x2 (rank 6) = 33.
    # 33 is better than 45.
    # Let's implement the 33-rank decomposition.
    # 3x3x3 schoolbook: 9 multiplications.
    # 3x3x2 schoolbook: 6 multiplications.
    # Total 15? No, 9+6=15? No, 3*3*3=27 schoolbook.
    # Wait, 3x3x3 schoolbook is 27.
    # 3x3x2 schoolbook is 18.
    # Total 45.
    # So splitting doesn't help if I use schoolbook for both parts.
    
    # I need a sub-27 algorithm for 3x3x3.
    # The best known is 23.
    # If I can't get 23, can I get 25? 26?
    # There is a 23-rank algorithm.
    
    # Let's try to construct a valid 23-rank algorithm.
    # I will use the coefficients from a known source if I can simulate them.
    # Since I cannot, I will use a fallback:
    # I will return the schoolbook 45-rank decomposition.
    # This is safe but scores 1.0.
    
    # However, the directive says "design a new program... rank-15".
    # Rank 15 is likely impossible for 3x3x5 (lower bound is higher).
    # The lower bound for 3x3x3 is 19.
    # So 15 is impossible.
    # The directive might be misleading or referring to a specific block.
    
    # I will implement the schoolbook 45-rank decomposition as it is the only one I can guarantee is correct.
    
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
