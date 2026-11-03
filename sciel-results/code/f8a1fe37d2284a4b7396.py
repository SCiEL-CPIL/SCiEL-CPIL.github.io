"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    # Build the target tensor as a list of (i,j,k) triples that need to be covered
    # T[i*m+j, j*p+k, k*n+i] = 1
    needed = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                needed.add((a, b, c))
    
    # Schoolbook solution as fallback
    def schoolbook():
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
    
    # Try to find a lower rank decomposition
    # A rank-1 term is (U_vec, V_vec, W_vec) where each is a sparse vector
    # For the term to be valid, the set of (a,b,c) covered must be a subset of needed
    
    # Strategy: try ranks from 40 down to 48, search for valid decompositions
    # Use random search with local improvements
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Precompute: for each pair (a,b), what c values are valid?
    # And for each (a,c), what b values?
    valid_ab_to_c = {}
    valid_ac_to_b = {}
    valid_bc_to_a = {}
    for (a, b, c) in needed:
        valid_ab_to_c.setdefault((a, b), set()).add(c)
        valid_ac_to_b.setdefault((a, c), set()).add(b)
        valid_bc_to_a.setdefault((b, c), set()).add(a)
    
    def is_valid_term(U_nonzero, V_nonzero, W_nonzero):
        """Check if a rank-1 term with given non-zero positions is valid."""
        for a in U_nonzero:
            for b in V_nonzero:
                for c in W_nonzero:
                    if (a, b, c) not in needed:
                        return False
        return True
    
    # Random search for rank r decomposition
    def try_rank(r, max_iters=50000):
        if time.time() > deadline:
            return None
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Generate r random rank-1 terms
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            covered = set()
            valid = True
            
            for c in range(r):
                # Pick random non-zero positions for this term
                # Each term should cover at least one needed entry
                # Try small support sizes
                
                # Randomly pick a needed entry to anchor this term
                a, b, k_val = random.choice(list(needed))
                
                # Try support size 1 first (simplest)
                U[a][c] = 1
                V[b][c] = 1
                W[k_val][c] = 1
                covered.add((a, b, k_val))
            
            # Check if all needed entries are covered
            if covered == needed:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try progressively lower ranks
    for r in range(40, min(48, rank_schoolbook)):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=20000)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
