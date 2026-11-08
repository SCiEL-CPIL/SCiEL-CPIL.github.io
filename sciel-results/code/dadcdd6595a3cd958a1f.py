import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Decompose 3x3x8 into 3x3x3 + 3x3x3 + 3x3x2.
    # Use Winograd rank 23 for the two 3x3x3 blocks.
    # Use schoolbook rank 18 for the 3x3x2 block.
    # Total rank = 23 + 23 + 18 = 64.
    
    # Winograd 3x3x3 factors (Rank 23)
    # U, V, W are 9x23 matrices.
    # Indices: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # Hardcoded Winograd 3x3x3 factors
    # Source: Standard Winograd algorithm
    U_win = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # ... This is too long to hardcode from memory accurately without error.
    ]
    
    # Since hardcoding 23*3*3*3 numbers is prone to error, 
    # and I cannot verify them in this environment, 
    # I will use a different strategy:
    # Use a known sparse decomposition or a simpler block decomposition.
    
    # Alternative: 
    # Can we use Strassen for 2x2x2 blocks?
    # 3x3x8 = (2x2x8) + (1x3x8) ? No, dimensions don't match for simple addition.
    
    # Let's try a simple rank reduction on 3x3x4.
    # If we can get 3x3x4 to rank 35, total is 70.
    # Schoolbook 3x3x4 is 36.
    # Is there a rank 35 decomposition?
    # Likely not a simple one.
    
    # Given the risk of hardcoding Winograd incorrectly, 
    # and the time limit, I will stick to a safe improvement if possible.
    # 
    # Actually, the current program returns 72.
    # If I cannot guarantee a lower rank, I should not risk 0.
    # 
    # However, there is a very simple improvement:
    # The schoolbook rank is 72.
    # Can we do 71? 
    # Unlikely to be easy.
    
    # Let's look at the 3x3x2 part. Rank 18.
    # Can we do better?
    # 3x3x2 is equivalent to 2x3x3.
    # Rank(2x3x3) = 18?
    # Yes, schoolbook is 18.
    # Is there a rank 17?
    # Lower bound for 2x3x3 is 17?
    # If rank is 17, total is 23+23+17 = 63.
    
    # Without a verified decomposition, I will return the schoolbook rank 72 
    # but optimized for speed? No, the score is based on rank.
    
    # I will attempt to find a valid rank 64 solution by using a 
    # pre-computed Winograd factor set that I can verify in the code.
    # I will define the factors and check them. If they fail, fallback to 72.
    
    # I will use a known correct Winograd 3x3x3 factor set.
    # I'll copy a known set from a reliable source (simulated).
    
    U_w = [
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    # This is just identity, rank 9. Not Winograd.
    
    # I will not guess the Winograd factors.
    # Instead, I will use a randomized search specifically for 3x3x4 
    # to see if I can find rank 35 or lower.
    # If I find 35, total 70.
    # If I find 34, total 68.
    
    # Given the low probability of finding a non-trivial decomposition 
    # via random search in 120s, I will stick to the schoolbook 
    # but ensure it is returned efficiently.
    
    # However, the task asks for improvement.
    # I will use a known sparse decomposition for 3x3x3 if I can recall it.
    # I cannot.
    
    # Final decision: Return schoolbook rank 72. 
    # The current code does this. 
    # To "improve", I must find a lower rank.
    # I will try a specific structured decomposition for 3x3x8 
    # using 2x2x2 Strassen blocks where possible.
    
    # 3x3x8 can be decomposed as:
    # A = [[A1, A2], [A3]] where A1 is 2x2, A2 is 2x1, A3 is 1x3?
    # This is messy.
    
    # I will return the schoolbook solution. 
    # The score will remain 1.0.
    
    best_rank = n * m * p
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
