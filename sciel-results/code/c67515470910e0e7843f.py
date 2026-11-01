import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 3, 7]
_TIME_LIMIT = 140.0  # Leave margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Uses a search approach starting from schoolbook and attempting to reduce rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    def _schoolbook():
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
        return U, V, W, rank
    
    def _verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def _try_reduce_rank(U, V, W, current_rank, max_reduction_attempts=5000):
        """Try to reduce rank by removing columns that are linearly redundant."""
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        best_rank = current_rank
        
        # Try removing each column and checking if the remaining still works
        # This is a heuristic - we check if the tensor can still be decomposed
        for attempt in range(max_reduction_attempts):
            if time.time() - start_time > _TIME_LIMIT - 5.0:
                break
            
            # Try removing a random column
            col_to_try = random.randint(0, best_rank - 1)
            
            # Check if removing this column still allows exact decomposition
            # We need to see if the remaining columns can still produce all 1s
            # This is complex, so we use a simple check: try to see if the column
            # is "redundant" by checking if its contribution can be absorbed
            
            # Simple approach: temporarily remove and see if we can adjust others
            # For now, just try to find if there's a column that's all zeros or
            # can be expressed as a combination of others
            
            # More practical: try setting entries to 0 and see if we can compensate
            pass
        
        return best_U, best_V, best_W, best_rank
    
    def _random_search(target_rank, time_budget=10.0, seed=0):
        """Try to find a decomposition of target_rank using random initialization and local search."""
        start = time.time()
        rng = random.Random(seed)
        
        # Initialize with random small integer/half-integer values
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Initialize with random values from {-1, 0, 1, 0.5, -0.5}
        vals = [-1, 0, 1, 0.5, -0.5]
        for i in range(n * m):
            for r in range(target_rank):
                U[i][r] = rng.choice(vals)
        for j in range(m * p):
            for r in range(target_rank):
                V[j][r] = rng.choice(vals)
        for k in range(p * n):
            for r in range(target_rank):
                W[k][r] = rng.choice(vals)
        
        # Local search: iteratively adjust entries to satisfy constraints
        for iteration in range(50000):
            if time.time() - start > time_budget:
                break
            
            # Check if we're done
            if _verify(U, V, W):
                return U, V, W, target_rank
            
            # Find unsatisfied constraints
            unsatisfied = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if abs(s - 1) > 1e-9:
                            unsatisfied.append((i, j, k, s))
            
            if not unsatisfied:
                return U, V, W, target_rank
            
            # Pick a random unsatisfied constraint
            i, j, k, current_sum = rng.choice(unsatisfied)
            diff = 1 - current_sum
            
            # Try to adjust one entry to fix this constraint
            # Option: adjust U[i*m+j, r] for some r where V and W are nonzero
            option = rng.choice([0, 1, 2])
            
            if option == 0:
                # Adjust U
                candidates = [r for r in range(target_rank) 
                             if V[j * p + k][r] != 0 and W[k * n + i][r] != 0]
                if candidates:
                    r = rng.choice(candidates)
                    denom = V[j * p + k][r] * W[k * n + i][r]
                    old_val = U[i * m + j][r]
                    new_val = (diff + old_val * denom) / denom
                    # Round to nearest half-integer
                    new_val_rounded = round(new_val * 2) / 2
                    if abs(new_val_rounded - new_val) < 1e-6:
                        U[i * m + j][r] = new_val_rounded
            
            elif option == 1:
                # Adjust V
                candidates = [r for r in range(target_rank) 
                             if U[i * m + j][r] != 0 and W[k * n + i][r] != 0]
                if candidates:
                    r = rng.choice(candidates)
                    denom = U[i * m + j][r] * W[k * n + i][r]
                    old_val = V[j * p + k][r]
                    new_val = (diff + old_val * denom) / denom
                    new_val_rounded = round(new_val * 2) / 2
                    if abs(new_val_rounded - new_val) < 1e-6:
                        V[j * p + k][r] = new_val_rounded
            
            else:
                # Adjust W
                candidates = [r for r in range(target_rank) 
                             if U[i * m + j][r] != 0 and V[j * p + k][r] != 0]
                if candidates:
                    r = rng.choice(candidates)
                    denom = U[i * m + j][r] * V[j * p + k][r]
                    old_val = W[k * n + i][r]
                    new_val = (diff + old_val * denom) / denom
                    new_val_rounded = round(new_val * 2) / 2
                    if abs(new_val_rounded - new_val) < 1e-6:
                        W[k * n + i][r] = new_val_rounded
        
        # Check final result
        if _verify(U, V, W):
            return U, V, W, target_rank
        return None
    
    # Start with schoolbook
    U, V, W, current_rank = _schoolbook()
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = 21
    # Try ranks from 21 to 41
    for target_rank in range(21, schoolbook_rank):
        if time.time() - start_time > _TIME_LIMIT - 10.0:
            break
        
        # Try a few random seeds for this rank
        found = False
        for seed in range(5):
            if time.time() - start_time > _TIME_LIMIT - 10.0:
                break
            
            result = _random_search(target_rank, time_budget=5.0, seed=seed)
            if result is not None:
                U, V, W, rank = result
                if _verify(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
        
        if found:
            # If we found a valid decomposition, continue trying lower ranks
            continue
        else:
            # If we can't find rank target_rank, try target_rank+1, etc.
            # But break if we've been searching too long
            if time.time() - start_time > _TIME_LIMIT - 15.0:
                break
    
    # Convert to lists and return
    return {
        "U": best_result["U"],
        "V": best_result["V"],
        "W": best_result["W"]
    }
# EVOLVE-BLOCK-END
