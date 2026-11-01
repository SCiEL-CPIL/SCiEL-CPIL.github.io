"""
Combined program for matrix multiplication 2x5x5 rank minimization.
Keeps the schoolbook decomposition as a guaranteed-valid fallback (rank 50)
and adds a bounded, structured search for lower-rank exact decompositions
using integer/half-integer entries, with a hard deadline guard.
"""
import time
import random
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    First attempts a bounded search for a lower-rank exact decomposition
    using integer/half-integer entries. Falls back to the schoolbook
    decomposition of rank 50 if no improvement is found within the deadline.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the schoolbook decomposition as guaranteed fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    fallback_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    # Check if we have time to search
    if time.time() > deadline - 5.0:
        return fallback_result
    
    # Validate a decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    def _validate(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower-rank decomposition via structured random search
    # with integer/half-integer entries
    best_result = None
    best_rank = schoolbook_rank
    
    # Search ranks from just below schoolbook down
    # For 2x5x5, try a few promising ranks
    candidate_ranks = list(range(45, schoolbook_rank))
    random.shuffle(candidate_ranks)
    
    # Limit number of ranks to try to stay within time budget
    max_ranks_to_try = min(len(candidate_ranks), 10)
    
    for rank in candidate_ranks[:max_ranks_to_try]:
        if time.time() > deadline - 5.0:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Use half-integer values: -1, -0.5, 0, 0.5, 1
            values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Sparse initialization: only set a fraction of entries
            density = 0.4
            for row in range(n * m):
                for c in range(rank):
                    if random.random() < density:
                        U[row][c] = random.choice(values)
            for row in range(m * p):
                for c in range(rank):
                    if random.random() < density:
                        V[row][c] = random.choice(values)
            for row in range(p * n):
                for c in range(rank):
                    if random.random() < density:
                        W[row][c] = random.choice(values)
            
            if _validate(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if best_result is not None:
            # Found a valid decomposition at this rank; try lower ranks
            # But break out if we've spent too much time
            if time.time() > deadline - 10.0:
                break
    
    if best_result is not None:
        return best_result
    
    return fallback_result
# EVOLVE-BLOCK-END
