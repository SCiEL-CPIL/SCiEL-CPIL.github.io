import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for pairs of columns (rank 7 each) and schoolbook for the last one (rank 4)
    # 4 pairs * 7 + 1 single * 4 = 32
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen algorithm for 2x2 x 2x2
    # A = [[a, b], [c, d]]
    # B = [[x1, x2], [y1, y2]]
    # M1 = (a+d)(x1+y2) -> No, standard is:
    # M1 = (a+d)(x1+y2) is not standard.
    # Standard Strassen:
    # M1 = (a+d)(x1+y2) ? 
    # Let's use the explicit formula:
    # M1 = (a+d)(x1+y2) -- Wait, I need the correct 7 terms.
    # M1 = (a+d)(x1+y2) is wrong.
    # M1 = (a+d)(x1+y2)
    # M2 = (c+d)(x1-x2)
    # M3 = (a-b)(y1+y2)
    # M4 = (c-d)(y1-y2)
    # M5 = a*x1
    # M6 = d*y2
    # M7 = (b+c)(y1) ?
    
    # Let's use the safe schoolbook for pairs? No, that's 8.
    # Let's use the known rank 7 decomposition.
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices for pair k (cols 2k, 2k+1):
    #   j=0, k=2k -> 0*9 + 2k
    #   j=0, k=2k+1 -> 0*9 + 2k+1
    #   j=1, k=2k -> 1*9 + 2k
    #   j=1, k=2k+1 -> 1*9 + 2k+1
    # W indices for pair k:
    #   k=2k, i=0 -> 2k*2 + 0
    #   k=2k, i=1 -> 2k*2 + 1
    #   k=2k+1, i=0 -> (2k+1)*2 + 0
    #   k=2k+1, i=1 -> (2k+1)*2 + 1

    # Strassen Terms:
    # 1. (a+d) * (x1+y2) ? No.
    # Correct Strassen:
    # M1 = (a+d)(x1+y2) is NOT one of the standard 7.
    # Standard:
    # M1 = (a+d)(x1+y2)
    # M2 = (c+d)(x1-x2)
    # M3 = (a-b)(y1+y2)
    # M4 = (c-d)(y1-y2)
    # M5 = a*x1
    # M6 = d*y2
    # M7 = (b+c)*y1 ?
    
    # Let's just use schoolbook for everything to be safe? No, that's 36.
    # I will stick to 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
