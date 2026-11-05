from fractions import Fraction
import time

def construct():
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Precompute reciprocal fractions
    recips = [Fraction(1, d) for d in denoms]
    
    # Precompute suffix sums
    # suffix[i] = sum(recips[i:])
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[i]
        
    # Greedy initialization
    best_sum = Fraction(0, 1)
    best_indices = []
    curr_sum = Fraction(0, 1)
    curr_indices = []
    
    for i in range(n):
        if curr_sum + recips[i] < 1:
            curr_sum += recips[i]
            curr_indices.append(i)
            
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_indices = curr_indices[:]
        
    # Branch and Bound
    # State: idx, curr_sum, curr_indices (passed by ref)
    start_time = time.time()
    deadline = 50.0 # seconds
    
    # We need to pass curr_indices carefully to avoid copying overhead
    # But Python list append/pop is O(1).
    
    def search(idx, curr_sum, curr_indices):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return
        
        # Prune: if current sum + max possible remaining <= best, stop
        if curr_sum + suffix[idx] <= best_sum:
            return
            
        # If current sum is valid, update best
        # Note: curr_sum is always < 1 due to pruning in include step? 
        # No, we check validity before adding.
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_indices = curr_indices[:]
            
        if idx == n:
            return
            
        # Option 1: Include denoms[idx]
        next_sum = curr_sum + recips[idx]
        if next_sum < 1:
            curr_indices.append(idx)
            search(idx + 1, next_sum, curr_indices)
            curr_indices.pop()
            
        # Option 2: Exclude denoms[idx]
        search(idx + 1, curr_sum, curr_indices)
        
    search(0, Fraction(0, 1), [])
    
    return best_indices
