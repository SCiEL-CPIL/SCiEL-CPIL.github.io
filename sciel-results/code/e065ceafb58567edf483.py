"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_deadline = time.time() + 140  # 140s out of 150s limit
    
    # First, try to find a decomposition with Strassen-based approach for the 3x3x8 case
    # The matrix multiplication tensor for (3,3,8) can be decomposed using
    # Strassen-like techniques on sub-blocks
    
    # Schoolbook rank is 3*3*8 = 72
    # Let's try to find a better decomposition
    
    # Approach: Use the fact that 3x3x8 can be split into two 3x3x4 blocks
    # For each 3x3x4 block, we can try to use a rank-14 decomposition (improving over schoolbook 12)
    # Actually, let's think about this differently.
    
    # The tensor T[i,j,k] = delta_{i,k} for the multiplication tensor
    # We need to find U, V, W such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Let's try a structured approach:
    # Split the 8 columns of B into groups and use Strassen on sub-problems
    
    # Actually, let me try a direct search for a lower rank decomposition.
    # Start with rank 64 and try to verify if it works.
    
    # For now, let's construct a valid decomposition using a hybrid approach:
    # Use 4x4 Strassen-like blocks where possible
    
    # Let me construct the schoolbook decomposition first as a baseline
    rank_school = n * m * p  # 72
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to find a better decomposition using Strassen-like techniques
    # For 3x3x3, the Strassen rank is 7 instead of 9
    # For 3x3x8, we can potentially use a combination
    
    # Let's try a rank-60 decomposition by combining Strassen on overlapping blocks
    
    # Actually, let me implement a proper search
    # The key insight: we can use the 3x3x3 Strassen decomposition (rank 7) 
    # and extend it to 3x3x8
    
    # For 3x3x3, Strassen decomposition:
    # The 7 rank-1 terms correspond to:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    
    # Let me try to construct a decomposition for 3x3x8 with rank < 72
    
    # Strategy: Partition the 8 columns into groups
    # Group 1: columns 0-3 (4 columns)
    # Group 2: columns 4-7 (4 columns)
    
    # For each 3x3x4 sub-problem, try to find a rank-14 decomposition
    # (schoolbook would be 3*3*4=36, so 14 is a big improvement if achievable)
    
    # Actually, let me try a simpler approach: use rank 64 by sharing some terms
    
    # Let me just return the schoolbook for now and ensure correctness
    # The evaluator will check if it's valid
    
    # Convert to lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
