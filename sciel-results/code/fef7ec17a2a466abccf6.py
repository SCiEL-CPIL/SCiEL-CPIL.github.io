"""Program: rank-minimal decomposition of M_{2,4,7} using linear algebraic compression."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7
    # We'll build the decomposition using a combination of Strassen-like techniques
    # on the 4-dimensional middle space.
    
    # Strategy: Use the identity C = A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x7.
    # Then apply Strassen-like combinations to reduce the number of 2x2*2x7 products.
    
    # For M_{2,2,7}, we can achieve rank 14 by using:
    # For a 2x2 matrix S and 2x7 matrix T, compute S*T using 14 scalar multiplications.
    # The trick: group the 7 columns into pairs and use Strassen on each pair, 
    # plus handle the last column separately.
    # 7 = 3*2 + 1, so 3 Strassen products (rank 7 each) + 1 naive product (rank 4) = 21? No.
    # Actually, for 2x2 * 2x2, Strassen uses 7 products.
    # For 2x2 * 2x7, we can view it as 3 instances of 2x2*2x2 plus one 2x2*2x1.
    # 3*7 + 4 = 21+4 = 25. Still not great.
    
    # Let me try a different approach. I'll use a direct construction.
    # The key insight from the directive: use cross-term cancellations.
    
    # Let me define the decomposition explicitly with rank 50.
    # I'll use the following structure:
    # - 7 Strassen products for the first 2x2*2x7 block
    # - 7 Strassen products for the second 2x2*2x7 block  
    # - Plus additional terms for the cross-block interactions
    # This gives 14 + 36 = 50? Let me think...
    
    # Actually, let me just implement a verified 56-rank decomposition first,
    # then try to optimize.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 28 x 56
    W = [[0] * rank for _ in range(p * n)]   # 14 x 56
    
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
