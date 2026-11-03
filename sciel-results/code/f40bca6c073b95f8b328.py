import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a rank-49 decomposition using Strassen-based block decomposition
    # Split 5 = 2 + 3
    # Use 7 Strassen products, each of which is a smaller matrix multiplication
    
    # The 7 Strassen products for blocks (2,3):
    # We need to handle the size mismatch by embedding
    # Actually, let's use a different known construction
    
    # Known rank-49 construction for 5x5x5:
    # This uses the fact that we can decompose the tensor using
    # algebraic identities over the blocks
    
    # Let me build it explicitly. The indices are:
    # U index: i*m + j = 5*i + j, where i,j in 0..4
    # V index: j*p + k = 5*j + k, where j,k in 0..4  
    # W index: k*n + i = 5*k + i, where k,i in 0..4
    
    # I'll use a verified construction with 49 terms
    
    # Construction based on block decomposition:
    # Partition {0,1,2,3,4} into S1={0,1}, S2={2,3,4}
    # For each of the 7 Strassen combinations, we get sub-problems
    
    # Let me just return the schoolbook for now and try to optimize
    # within the time budget
    
    # Fallback: schoolbook rank 125
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(N)]
    V_school = [[0] * rank_school for _ in range(M)]
    W_school = [[0] * rank_school for _ in range(P)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank exact decomposition
    # Use a greedy approach: start with schoolbook, try to express columns as sums
    
    # Actually, let me try a direct construction with rank 49
    # The construction uses the following 49 rank-1 tensors
    
    # I'll generate them based on block decomposition principles
    # For now, return schoolbook as safe fallback
    
    if time.time() > deadline - 5:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to verify if a rank-49 solution exists by construction
    # Use the known algebraic identity
    
    # For 5x5, partition into 2x2, 2x3, 3x2, 3x3 blocks
    # Strassen gives 7 products instead of 8
    # But the products have different dimensions, so we need to handle them
    
    # M1 = (A11+A22)(B11+B22): sizes don't match for direct addition
    # We need to embed: pad A11 to 3x3 and A22 stays 3x3, or vice versa
    
    # Let me try a different known construction
    
    # Actually, the simplest verified improvement: use the fact that
    # the multiplication tensor has symmetries we can exploit
    
    # For now, return the schoolbook solution which is guaranteed valid
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
