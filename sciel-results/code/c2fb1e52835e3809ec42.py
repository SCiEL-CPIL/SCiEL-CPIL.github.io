import time
import random
import numpy as np
from itertools import combinations

N = 512
TIME_LIMIT = 80  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute for each position x, the list of (d, positions) tuples
    # that would form a 4-AP if x is at a given position
    # For speed, precompute all 4-APs and build a conflict structure
    
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0, all in [1, 512]
    # For each element x, store which triples of other elements would complete a 4-AP with x
    
    # Precompute: for each x, list of (other1, other2, other3) such that
    # {x, other1, other2, other3} forms a 4-AP
    
    # This is expensive to precompute fully, so let's use a smarter approach:
    # For each x, precompute the set of "forbidden pairs" - if two of the other
    # 3 elements are in the set, then x cannot be added.
    
    # Actually, the cleanest fast approach: represent the set as a bitset (numpy array)
    # and use vectorized checks.
    
    def can_add_vectorized(x, current_bitset):
        """Check if adding x creates a 4-term AP using vectorized numpy ops."""
        # Position 0: x, x+d, x+2d, x+3d for d in 1..(N-x)//3
        max_d = (N - x) // 3
        if max_d > 0:
            ds = np.arange(1, max_d + 1)
            # Check if all three: x+d, x+2d, x+3d are in current_bitset
            # current_bitset is a boolean array of length N+1 (index 1..N)
            mask = current_bitset[x+ds] & current_bitset[x+2*ds] & current_bitset[x+3*ds]
            if np.any(mask):
                return False
        
        # Position 1: x-d, x, x+d, x+2d for d in 1..min(x-1, (N-x)//2)
        max_d = min(x - 1, (N - x) // 2)
        if max_d > 0:
            ds = np.arange(1, max_d + 1)
            mask = current_bitset[x-ds] & current_bitset[x+ds] & current_bitset[x+2*ds]
            if np.any(mask):
                return False
        
        # Position 2: x-2d, x-d, x, x+d for d in 1..min((x-1)//2, N-x)
        max_d = min((x - 1) // 2, N - x)
        if max_d > 0:
            ds = np.arange(1, max_d + 1)
            mask = current_bitset[x-2*ds] & current_bitset[x-ds] & current_bitset[x+ds]
            if np.any(mask):
                return False
        
        # Position 3: x-3d, x-2d, x-d, x for d in 1..(x-1)//3
        max_d = (x - 1) // 3
        if max_d > 0:
            ds = np.arange(1, max_d + 1)
            mask = current_bitset[x-3*ds] & current_bitset[x-2*ds] & current_bitset[x-ds]
            if np.any(mask):
                return False
        
        return True
    
    def greedy_randomized(current_bitset=None):
        """Greedy with random ordering, using bitset for speed."""
        if current_bitset is None:
            current_bitset = np.zeros(N + 1, dtype=bool)
        else:
            current_bitset = current_bitset.copy()
        
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if not current_bitset[x]:
                if can_add_vectorized(x, current_bitset):
                    current_bitset[x] = True
        return current_bitset
    
    def local_search(current_bitset):
        """Improve the set by adding elements and trying swaps."""
        current_bitset = current_bitset.copy()
        improved = True
        
        while improved and time.time() - start_time < TIME_LIMIT - 2:
            improved = False
            
            # Phase 1: Try to add any element that can be added
            for x in range(1, N + 1):
                if not current_bitset[x]:
                    if can_add_vectorized(x, current_bitset):
                        current_bitset[x] = True
                        improved = True
        
        # Phase 2: Try swaps - remove one element, see if we can add more
        for remove_x in range(1, N + 1):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            if not current_bitset[remove_x]:
                continue
            
            trial_bitset = current_bitset.copy()
            trial_bitset[remove_x] = False
            
            # Try to add as many elements as possible
            added_count = 0
            for add_x in range(1, N + 1):
                if not trial_bitset[add_x]:
                    if can_add_vectorized(add_x, trial_bitset):
                        trial_bitset[add_x] = True
                        added_count += 1
            
            if added_count >= 1:
                # Net gain: we removed 1, added added_count
                # Only accept if net positive
                if added_count >= 1:
                    current_bitset = trial_bitset
                    improved = True
                    break  # Restart the outer loop
        
        return current_bitset
    
    def bitset_to_set(bitset):
        """Convert numpy bitset to a Python set."""
        return set(int(i) for i in np.where(bitset)[0] if i > 0)
    
    best_bitset = np.zeros(N + 1, dtype=bool)
    best_size = 0
    
    # Run multiple randomized greedy restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        current_bitset = greedy_randomized()
        current_bitset = local_search(current_bitset)
        
        size = int(np.sum(current_bitset))
        if size > best_size:
            best_size = size
            best_bitset = current_bitset.copy()
    
    # Try base-3 construction as another starting point
    if time.time() - start_time < TIME_LIMIT - 5:
        base3_bitset = np.zeros(N + 1, dtype=bool)
        for i in range(1, N + 1):
            n = i
            valid = True
            while n > 0:
                if n % 3 == 2:
                    valid = False
                    break
                n //= 3
            if valid:
                base3_bitset[i] = True
        
        base3_size = int(np.sum(base3_bitset))
        if base3_size > best_size:
            best_bitset = base3_bitset.copy()
            best_size = base3_size
        
        # Local search on base-3
        improved_bitset = local_search(base3_bitset)
        improved_size = int(np.sum(improved_bitset))
        if improved_size > best_size:
            best_bitset = improved_bitset
            best_size = improved_size
    
    # Convert to sorted list
    result = sorted(int(i) for i in np.where(best_bitset)[0] if i > 0)
    
    # If somehow empty (shouldn't happen), return a valid singleton
    if not result:
        result = [1]
    
    return result
