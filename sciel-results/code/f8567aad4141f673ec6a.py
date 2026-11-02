"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check exact decomposition:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    for all i, k, j1, j2
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0
                    for r in range(rank):
                        s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank_gf(n, m, p, target_rank, deadline, seed):
    """
    Attempt to find a decomposition of rank 'target_rank' using random search over GF(3).
    We map solutions found in GF(3) to integers.
    """
    rng = random.Random(seed)
    max_iters = 2000
    
    # Pre-compute indices for constraints
    # Constraint: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1, j2)
    # Variables: U, V, W entries in GF(3)
    
    for attempt in range(max_iters):
        if time.time() > deadline:
            return None
            
        # Initialize with random values in GF(3)
        U = [[rng.randint(0, 2) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.randint(0, 2) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.randint(0, 2) for _ in range(target_rank)] for _ in range(p*n)]
        
        # Check all constraints in GF(3)
        valid = True
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(target_rank):
                            s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                        s %= 3
                        target = 1 if j1 == j2 else 0
                        if s != target:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to integers for the final output
            # The problem allows integers. GF(3) values {0, 1, 2} are integers.
            # Note: The decomposition is exact over integers if it is exact over GF(3)?
            # Yes, because the operations are linear and the target is 0/1.
            # If sum(...) = 1 mod 3, it could be 1, 4, -2, etc.
            # However, if we find a solution with small entries, it's likely the exact integer solution.
            # To be safe, we verify over integers.
            if _verify(U, V, W, n, m, p):
                return U, V, W
                
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook (rank 18)
    U_best, V_best, W_best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Search for lower ranks
    # Lower bound is 9.
    # We will try ranks from 17 down to 9.
    
    seed = 42
    for target_rank in range(best_rank - 1, 8, -1):
        if time.time() > deadline - 5:
            break
            
        res = _try_rank_gf(n, m, p, target_rank, deadline, seed)
        if res:
            U_best, V_best, W_best = res
            best_rank = target_rank
            # Continue searching for even lower ranks?
            # Yes, but we break the loop if we find a lower rank and update best_rank?
            # The loop goes downwards. If we find 17, we continue to 16.
            # But we should probably stop if we find a solution? No, we want minimal.
            # So we continue the loop.
            # But we need to make sure we don't miss the chance to find 16 if we found 17.
            # The loop structure handles this.
        
        seed += 1
        
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
