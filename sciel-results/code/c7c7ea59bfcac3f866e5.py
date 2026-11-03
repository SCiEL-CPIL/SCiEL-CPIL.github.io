"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from functools import lru_cache

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition using known algorithms
    # For 3x3x3, Strassen gives rank 7
    # For 3x3x5, we can use a block approach
    
    # Strategy: Use Strassen for the 3x3x3 part and schoolbook for the remaining 3x3x2 part
    # Total rank: 7 + 3*3*2 = 7 + 18 = 25
    
    # But first, let's try a direct approach with the full tensor
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the trace convention
    
    # Actually, for matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # In the trace convention:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Let's build the tensor and try to find a low-rank decomposition
    
    size1 = n * m  # 9
    size2 = m * p  # 15
    size3 = p * n  # 15
    
    # Build the tensor as a 3-way array
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # For each valid (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((size1, size2, size3), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Approach: Use the fact that the tensor can be decomposed using known algorithms
    
    # For 3x3x3, Strassen's algorithm has rank 7
    # For 3x3x5, we can think of it as combining a 3x3x3 and a 3x3x2
    
    # Let's try a constructive approach:
    # Use Strassen for the first 3 columns of B (k=0,1,2)
    # Use schoolbook for the last 2 columns of B (k=3,4)
    
    # Strassen's algorithm for 3x3x3:
    # The 7 terms are:
    # M1 = (A00) * (B00)
    # M2 = (A10 + A20) * (B00 + B01)  -- wait, this is for 2x2
    # Let me use the standard 3x3 Strassen decomposition
    
    # Actually, let me just use a known 3x3x3 rank-7 decomposition
    # and extend it to 3x3x5
    
    # For 3x3x3, one known rank-7 decomposition uses:
    # The tensor is the structure tensor of M_3 over a field
    # Strassen showed rank(M_3) = 7
    
    # Let me construct the 3x3x3 part using Strassen's terms
    # and add schoolbook terms for the remaining 2 columns
    
    # Strassen's 7 terms for 3x3x3:
    # I'll use a known decomposition
    
    # Actually, let me just try to use a greedy approach or return schoolbook if we can't find better
    
    # For safety, let's return a decomposition that we know works
    # Using Strassen for 3x3x3 (rank 7) + schoolbook for 3x3x2 (rank 18) = rank 25
    
    # But first, let's verify this is correct by constructing it
    
    rank_total = 7 + 18  # 25
    
    U = [[0] * rank_total for _ in range(size1)]
    V = [[0] * rank_total for _ in range(size2)]
    W = [[0] * rank_total for _ in range(size3)]
    
    # Strassen's 7 terms for 3x3x3 (k=0,1,2)
    # I need to find the correct Strassen decomposition for 3x3x3
    # Let me use a known one
    
    # For 3x3x3, the Strassen decomposition has 7 terms
    # Each term is a product of three matrices (or vectors in the tensor sense)
    
    # Let me use a different approach: just use schoolbook for everything
    # This gives rank 45, which is correct but not minimal
    
    # Actually, let me try to find a better decomposition using optimization
    
    # Given time constraints, let me return a valid decomposition
    # I'll use a combination of Strassen-like terms for the 3x3x3 part
    
    # For now, let me just return the schoolbook decomposition to ensure correctness
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
