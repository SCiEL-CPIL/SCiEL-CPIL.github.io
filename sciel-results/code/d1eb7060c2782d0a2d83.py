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
    
    # Try to find a lower-rank decomposition via structured pattern search
    # and random search with integer/half-integer entries
    best_result = None
    best_rank = schoolbook_rank
    
    # First: try structured pattern-based approaches that exploit symmetry
    # For 2x5x5, we can try decompositions based on Kronecker products or
    # structured patterns that reduce rank
    
    # Pattern 1: Try rank = n*p + m*p - 1 style decompositions (Strassen-like)
    # For 2x5x5, some known bounds suggest rank could be less than 50
    
    # Try a few specific structured patterns
    def _try_structured_pattern(rank):
        """Try a structured decomposition pattern for given rank."""
        if time.time() > deadline - 5.0:
            return None
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Pattern: Use first n*p entries for direct terms, rest for corrections
        col = 0
        # Direct terms for i=0
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[0 * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + 0][col] = 1.0
                    col += 1
        # Direct terms for i=1
        for j in range(m):
            for k in range(p):
                if col < rank:
                    U[1 * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + 1][col] = 1.0
                    col += 1
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try structured patterns for a few ranks
    for test_rank in [45, 40, 35, 30]:
        if time.time() > deadline - 5.0:
            break
        result = _try_structured_pattern(test_rank)
        if result is not None:
            best_result = result
            best_rank = test_rank
            break
    
    # Second: random search with better initialization
    if best_result is None or best_rank > 45:
        candidate_ranks = list(range(max(20, best_rank - 5), best_rank if best_rank < schoolbook_rank else schoolbook_rank))
        if not candidate_ranks:
            candidate_ranks = list(range(45, schoolbook_rank))
        random.shuffle(candidate_ranks)
        
        max_ranks_to_try = min(len(candidate_ranks), 8)
        
        for rank in candidate_ranks[:max_ranks_to_try]:
            if time.time() > deadline - 5.0:
                break
            
            max_attempts = 30
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                values = [-1.0, -0.5, 0.0, 0.5, 1.0]
                
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                # Adaptive density based on rank
                density = max(0.3, min(0.6, 50.0 / (rank * 3)))
                
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
                if time.time() > deadline - 10.0:
                    break
    
    if best_result is not None:
        return best_result
    
    return fallback_result
# EVOLVE-BLOCK-END
