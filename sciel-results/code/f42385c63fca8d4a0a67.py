import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    # Try to find a low rank decomposition for M_{3,3,2}
    def try_find_m332(rank_target, max_iters=200):
        # n=3, m=3, p=2
        n2, m2, p2 = 3, 3, 2
        
        for _ in range(max_iters):
            U = [[random.choice([-1, 0, 1]) for _ in range(rank_target)] for _ in range(n2 * m2)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank_target)] for _ in range(m2 * p2)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank_target)] for _ in range(p2 * n2)]
            
            # Check
            is_valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0
                        for r in range(rank_target):
                            s += U[i * m2 + j][r] * V[j * p2 + k][r] * W[k * n2 + i][r]
                        if s != 1:
                            is_valid = False
                            break
                    if not is_valid: break
                if not is_valid: break
            if is_valid:
                return U, V, W
        return None

    def combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """Combine two decompositions: M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}"""
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank_total = rank1 + rank2
        
        U_full = [[0] * rank_total for _ in range(n * m)]
        V_full = [[0] * rank_total for _ in range(m * (p1 + p2))]
        W_full = [[0] * rank_total for _ in range((p1 + p2) * n)]
        
        # Copy M_{n,m,p1} part
        for i in range(n * m):
            for r in range(rank1):
                U_full[i][r] = U1[i][r]
        
        for j in range(m):
            for k in range(p1):
                for r in range(rank1):
                    V_full[j * (p1 + p2) + k][r] = V1[j * p1 + k][r]
        
        for k in range(p1):
            for i in range(n):
                for r in range(rank1):
                    W_full[k * n + i][r] = W1[k * n + i][r]
        
        # Copy M_{n,m,p2} part
        for i in range(n * m):
            for r in range(rank2):
                U_full[i][rank1 + r] = U2[i][r]
        
        for j in range(m):
            for k in range(p2):
                for r in range(rank2):
                    V_full[j * (p1 + p2) + p1 + k][rank1 + r] = V2[j * p2 + k][r]
        
        for k in range(p2):
            for i in range(n):
                for r in range(rank2):
                    W_full[(p1 + k) * n + i][rank1 + r] = W2[k * n + i][r]
        
        return {"U": U_full, "V": V_full, "W": W_full}

    start_time = time.time()
    deadline = 100  # Safety margin
    
    # Start with schoolbook
    best_result = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: M_{3,3,5} = M_{3,3,3} + M_{3,3,2}
    # Schoolbook M_{3,3,3} has rank 27
    # Try to find M_{3,3,2} with rank < 18
    
    for r2 in range(10, 18):
        if time.time() - start_time > deadline:
            break
        
        max_iters = 50 if r2 <= 12 else 30
        res = try_find_m332(r2, max_iters=max_iters)
        
        if res:
            U2, V2, W2 = res
            
            # Get schoolbook M_{3,3,3}
            sb333 = schoolbook(3, 3, 3)
            U1, V1, W1 = sb333["U"], sb333["V"], sb333["W"]
            
            # Combine
            combined = combine_decompositions(U1, V1, W1, U2, V2, W2, 3, 3, 3, 2)
            
            # Verify
            if verify_decomposition(combined["U"], combined["V"], combined["W"], 3, 3, 5):
                total_rank = 27 + r2
                if total_rank < best_rank:
                    best_result = combined
                    best_rank = total_rank
                    break  # Found a valid improvement, stop
    
    return best_result
# EVOLVE-BLOCK-END
