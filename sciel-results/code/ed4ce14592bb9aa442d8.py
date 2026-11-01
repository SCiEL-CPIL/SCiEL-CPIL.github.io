"""Matrix multiplication 2x4 x 4x4 tensor decomposition - attempting rank 27."""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # First, build the rank-28 Strassen decomposition as a baseline
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28
        
        col = 0
        for b in range(2):  # B column block: 0 or 1
            for a_blk in range(2):  # A column block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    
                    # U side: A[i, 2*a_blk + j]
                    # U flat index: i * 4 + (2*a_blk + j)
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff
                    
                    # V side: B[2*a_blk + j, 2*b + k]
                    # V flat index: (2*a_blk + j) * 4 + (2*b + k)
                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff
                    
                    # W side: C[i, 2*b + k]
                    # W flat index: (2*b + k) * 2 + i
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b_idx in range(m*p):
                for c_idx in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b_idx // p
                    k1 = b_idx % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b_idx][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Try to find rank-27 decomposition
    # Strategy: Start from rank-28 Strassen and try to eliminate one column
    # by finding a linear combination that preserves the tensor structure
    
    best_result = build_strassen_28()
    best_rank = 28
    
    # Try local search: perturb the rank-28 decomposition and see if we can
    # reduce rank by finding a column that can be expressed as a combination
    # of others (linear dependency in the tensor space)
    
    # Alternative: Direct search for rank-27 with structured initialization
    # Use the Strassen structure as a starting point and try small perturbations
    
    for target_rank in [27]:
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + target_rank * 1000003)
            
            # Initialize with Strassen-28 and randomly zero out one column
            # then try to repair
            U28, V28, W28 = best_result["U"], best_result["V"], best_result["W"]
            
            # Pick a column to remove
            remove_col = random.randint(0, 27)
            
            U = [row[:remove_col] + row[remove_col+1:] for row in U28]
            V = [row[:remove_col] + row[remove_col+1:] for row in V28]
            W = [row[:remove_col] + row[remove_col+1:] for row in W28]
            
            # Now try to repair by adding small perturbations to make it valid
            # This is a hard problem, so instead try a different approach
            
            # Try: generate a random rank-27 decomposition with half-integer entries
            # biased towards the Strassen structure
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            # Start from Strassen-28, remove one column, then try to fix
            # by adjusting other columns slightly
            U = [row[:remove_col] + row[remove_col+1:] for row in U28]
            V = [row[:remove_col] + row[remove_col+1:] for row in V28]
            W = [row[:remove_col] + row[remove_col+1:] for row in W28]
            
            if validate(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            break
    
    # If rank 27 not found, try a more systematic search
    if best_rank > 27 and time.time() < deadline - 10:
        # Try structured search: the 2x4x4 tensor has 8*16*8 = 1024 entries
        # but only 32 are non-zero (the "diagonal" entries)
        # Rank 27 means 27*32 = 864 equations to satisfy with 27*(8+16+8) = 1296 variables
        # This is underdetermined, so solutions likely exist
        
        # Try random search with more attempts
        for attempt in range(2000):
            if time.time() > deadline:
                break
            
            random.seed(1234567 + attempt * 31)
            
            # Generate random half-integer matrices
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            U = [[rand_half_int() for _ in range(27)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(27)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(27)] for _ in range(p*n)]
            
            if validate(U, V, W, 27):
                best_rank = 27
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result

# EVOLVE-BLOCK-END
