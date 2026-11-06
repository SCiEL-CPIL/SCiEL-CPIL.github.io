import random
import time
from itertools import combinations

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_max_sidon_size(A, num_trials=10):
    """
    Estimates the maximum Sidon subset size using a greedy approach with multiple random restarts.
    This is much faster than exact search and provides a good lower bound.
    For minimization, we want this value to be small.
    """
    n = len(A)
    best_size = 0
    
    for _ in range(num_trials):
        # Shuffle A to randomize the greedy choice
        A_shuffled = A[:]
        random.shuffle(A_shuffled)
        
        current_subset = []
        sums = set()
        
        for x in A_shuffled:
            # Check if adding x creates a collision
            valid = True
            for y in current_subset:
                s = x + y
                if s in sums:
                    valid = False
                    break
            if valid:
                # Add x and update sums
                for y in current_subset:
                    sums.add(x + y)
                sums.add(2 * x) # Diagonal
                current_subset.append(x)
        
        if len(current_subset) > best_size:
            best_size = len(current_subset)
            
    return best_size

def _max_sidon_size(A):
    n = len(A)
    # Check from largest size down to 1
    # If we find a Sidon subset of size k, that is the maximum.
    for k in range(n, 0, -1):
        for comb in combinations(A, k):
            if _is_sidon(comb):
                return k
    return 0 # Should not happen for n>=1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    random.seed(42)
    
    best_set = None
    best_val = 100 # Initialize high
    
    # Initial candidates: Dense sets are likely to have small max Sidon subsets
    # Strategy: Start with consecutive integers or arithmetic progressions
    
    initial_candidates = []
    # Consecutive
    for start in range(0, 117):
        initial_candidates.append(list(range(start, start + 12)))
    # Arithmetic progressions with small step
    for step in [2, 3, 4]:
        for start in range(0, 129 - 11*step):
            initial_candidates.append(list(range(start, start + 12*step, step)))
    # Random dense sets
    for _ in range(50):
        # Pick a small range and sample from it
        range_size = random.randint(12, 30)
        start = random.randint(0, 128 - range_size)
        if range_size >= 12:
             initial_candidates.append(sorted(random.sample(range(start, start + range_size), 12)))

    # Evaluate initial candidates
    for cand in initial_candidates:
        # Use greedy for speed in initial screening
        val = _greedy_max_sidon_size(cand, num_trials=5)
        if val < best_val:
            # Verify with exact method if it looks promising (val <= 5)
            exact_val = _max_sidon_size(cand)
            if exact_val < best_val:
                best_val = exact_val
                best_set = cand[:]
                
    # If we found a very good candidate, we can try to improve it
    # Otherwise, run local search from the best candidate found
    
    if best_set is None:
        best_set = initial_candidates[0]
        best_val = _max_sidon_size(best_set)

    current_set = best_set[:]
    current_val = best_val
    
    # Local Search
    while time.time() < deadline:
        # Pick a random element to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        improved = False
        # Try a few random replacements, biased towards keeping density?
        # Or just random, but check with greedy first
        
        for _ in range(50): # Try 50 random moves
            # Generate new value
            # Bias towards values close to existing ones to maintain density
            if random.random() < 0.7:
                # Pick a value close to one of the existing elements
                ref_idx = random.randint(0, 11)
                ref_val = current_set[ref_idx]
                offset = random.randint(-5, 5)
                new_val_elem = ref_val + offset
            else:
                new_val_elem = random.randint(0, 128)
                
            if new_val_elem < 0 or new_val_elem > 128:
                continue
            if new_val_elem == old_val:
                continue
            if new_val_elem in current_set:
                continue
                
            # Perform swap
            current_set[idx] = new_val_elem
            current_set.sort()
            
            # Fast evaluation
            val = _greedy_max_sidon_size(current_set, num_trials=3)
            
            if val < current_val:
                # Potential improvement, verify with exact
                exact_val = _max_sidon_size(current_set)
                if exact_val < current_val:
                    current_val = exact_val
                    if current_val < best_val:
                        best_val = current_val
                        best_set = current_set[:]
                    improved = True
                    break
                else:
                    # Greedy said better, but exact says no (or same)
                    # Revert
                    current_set[idx] = old_val
                    current_set.sort()
            else:
                # Greedy says not better, revert
                current_set[idx] = old_val
                current_set.sort()
            
            if time.time() > deadline - 5:
                break

        if not improved:
            # Try a larger jump or restart from best
            if random.random() < 0.1:
                current_set = best_set[:]
                current_val = best_val

    return best_set
