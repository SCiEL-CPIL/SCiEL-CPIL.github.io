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
    
    Strategy: Exploit the bilinear structure by sharing intermediate products
    between the two output rows. Search for lower-rank 5x5 matrix multiplication
    decompositions and lift them, plus direct structured/random searches.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    if time.time() > deadline - 5.0:
        return fallback_result
    
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
    
    def _validate_5x5(U5, V5, W5, m, p):
        rank = len(U5[0])
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U5[j][c] * V5[j*p+k][c] * W5[k][c]
                if abs(s - 1.0) > 1e-9:
                    return False
        return True
    
    def _lift_5x5_to_2x5x5(U5, V5, W5, n, m, p):
        r = len(U5[0])
        total_rank = 2 * r
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for c in range(r):
            for j in range(m):
                U[0 * m + j][c] = U5[j][c]
            for j in range(m):
                for k in range(p):
                    V[j * p + k][c] = V5[j * p + k][c]
            for k in range(p):
                W[k * n + 0][c] = W5[k][c]
            for j in range(m):
                U[1 * m + j][c + r] = U5[j][c]
            for j in range(m):
                for k in range(p):
                    V[j * p + k][c + r] = V5[j * p + k][c]
            for k in range(p):
                W[k * n + 1][c + r] = W5[k][c]
        
        return {"U": U, "V": V, "W": W}
    
    def _search_5x5(rank, max_attempts=50):
        m5, p5 = 5, 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 10.0:
                return None
            values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U5 = [[0] * rank for _ in range(m5)]
            V5 = [[0] * rank for _ in range(m5 * p5)]
            W5 = [[0] * rank for _ in range(p5)]
            density = max(0.3, min(0.6, 25.0 / (rank * 3)))
            for row in range(m5):
                for c in range(rank):
                    if random.random() < density:
                        U5[row][c] = random.choice(values)
            for row in range(m5 * p5):
                for c in range(rank):
                    if random.random() < density:
                        V5[row][c] = random.choice(values)
            for row in range(p5):
                for c in range(rank):
                    if random.random() < density:
                        W5[row][c] = random.choice(values)
            if _validate_5x5(U5, V5, W5, m5, p5):
                return {"U5": U5, "V5": V5, "W5": W5}
        return None
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Phase 1: Try to find a 5x5 decomposition and lift it
    for r5 in range(24, 19, -1):
        if time.time() > deadline - 20.0:
            break
        result5 = _search_5x5(r5, max_attempts=30)
        if result5 is not None:
            lifted = _lift_5x5_to_2x5x5(result5["U5"], result5["V5"], result5["W5"], n, m, p)
            if _validate(lifted["U"], lifted["V"], lifted["W"], n, m, p):
                best_result = lifted
                best_rank = 2 * r5
                break
    
    # Phase 2: Direct random search on 2x5x5 tensor
    if best_result is None or best_rank > 45:
        candidate_ranks = list(range(48, max(38, best_rank - 1), -1))
        random.shuffle(candidate_ranks)
        
        for rank in candidate_ranks[:6]:
            if time.time() > deadline - 15.0:
                break
            for attempt in range(40):
                if time.time() > deadline - 10.0:
                    break
                values = [-1.0, -0.5, 0.0, 0.5, 1.0]
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                density = max(0.25, min(0.5, 50.0 / (rank * 3)))
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
