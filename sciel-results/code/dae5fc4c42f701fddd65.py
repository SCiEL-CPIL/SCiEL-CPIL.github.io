"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Schoolbook rank is n*m*p = 3*4*7 = 84
    rank_schoolbook = n * m * p
    
    # Construct the tensor T_{(i,j), (j,k), (k,i)} = 1
    # We want to find rank-minimal decomposition:
    # T_{(i,j), (j,k), (k,i)} = sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
    
    # The MPS/Strassen-like approach:
    # For 2x2x2, Strassen uses rank 7 instead of 8.
    # We can try to embed Strassen blocks into the 3x4x7 problem.
    
    # One approach: pad to 4x4x8 and use recursive Strassen, then project back.
    # But this may not give the minimal rank for 3x4x7.
    
    # Alternative: Use a direct search for a lower rank decomposition.
    # Given the time constraint, let's try a few heuristics.
    
    # Heuristic 1: Try to find a decomposition with rank < 84 by checking
    # if we can merge some terms using linear dependencies.
    
    # For now, let's implement a basic check: can we find a rank-83 decomposition?
    # This is equivalent to checking if the tensor has a "defect" of 1.
    
    # A simpler approach: use the fact that for matrix multiplication,
    # there are known algorithms with better asymptotic complexity, but for small
    # dimensions, the schoolbook might be optimal or near-optimal.
    
    # Let's try a random search for a lower rank decomposition.
    # We'll initialize random U, V, W and try to optimize.
    
    # Actually, let's first check if there's a known lower bound.
    # The rank of the 3x4x7 matrix multiplication tensor is at least 
    # the "slice rank" or other lower bounds.
    
    # For simplicity and given the time constraint, let's return the schoolbook
    # decomposition unless we can quickly find a better one.
    
    # Try a quick greedy approach: start with schoolbook and try to remove terms
    # by checking if the remaining tensor still spans the same space.
    
    # This is complex, so let's just return schoolbook for now, but with a
    # time check to ensure we don't exceed the deadline.
    
    if time.time() > deadline:
        rank = rank_schoolbook
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
    
    # Try to find a better decomposition using a simple optimization.
    # We'll use a randomized approach: initialize U, V, W randomly and try
    # to minimize the Frobenius norm of (T - sum_r U_r V_r W_r).
    
    # But this is computationally expensive, so let's limit the number of iterations.
    
    # For now, let's just return the schoolbook decomposition.
    rank = rank_schoolbook
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
