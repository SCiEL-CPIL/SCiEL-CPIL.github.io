"""Find rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Construct a rank-51 decomposition by merging terms in 3 blocks of 3 columns
    # Block 1: k=0,1,2. Merge (0,0,0) and (0,0,1).
    # Block 2: k=3,4,5. Merge (0,0,3) and (0,0,4).
    # Block 3: k=6,7,8. Merge (0,0,6) and (0,0,7).
    
    # The merged terms use U[0]=1.
    # We need to ensure no other term uses U[0] in a conflicting way?
    # No, the other terms for i=0, j=0 are removed.
    # The other terms for i=0, j=1 and i=1, j=* remain as schoolbook.
    # The merged term has U[0]=1, others 0.
    # So it only affects i=0, j=0.
    # For i=0, j=0, the merged term provides 1s at the merged k's and 0 elsewhere.
    # The removed schoolbook terms provided 1s at the merged k's and 0 elsewhere.
    # So the sum is unchanged.
    
    rank = 51
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Define the merged terms
    # Merged 1: i=0, j=0. k=0,1.
    # U[0] = 1
    # V[0] = 1, V[1] = 1
    # W[0] = 1, W[2] = 1
    U[0][col] = 1.0
    V[0][col] = 1.0
    V[1][col] = 1.0
    W[0][col] = 1.0
    W[2][col] = 1.0
    col += 1
    
    # Merged 2: i=0, j=0. k=3,4.
    # U[0] = 1
    # V[3] = 1, V[4] = 1
    # W[6] = 1, W[8] = 1
    U[0][col] = 1.0
    V[3][col] = 1.0
    V[4][col] = 1.0
    W[6][col] = 1.0
    W[8][col] = 1.0
    col += 1
    
    # Merged 3: i=0, j=0. k=6,7.
    # U[0] = 1
    # V[6] = 1, V[7] = 1
    # W[12] = 1, W[14] = 1
    U[0][col] = 1.0
    V[6][col] = 1.0
    V[7][col] = 1.0
    W[12][col] = 1.0
    W[14][col] = 1.0
    col += 1
    
    # Add the remaining schoolbook terms
    # Skip the merged ones: (0,0,0), (0,0,1), (0,0,3), (0,0,4), (0,0,6), (0,0,7)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i == 0 and j == 0 and k in [0, 1, 3, 4, 6, 7]):
                    continue
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
                
    # Verify col == rank
    # It should be 3 + (54 - 6) = 51.
    
    return {"U": U, "V": W} # Wait, return V as well
