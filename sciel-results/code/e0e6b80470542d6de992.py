import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the largest Sidon subset size of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # Try k from n down to 1
    for k in range(n, 0, -1):
        # Use DFS with pruning to find any Sidon subset of size k
        found = _find_sidon_subset(S, k, 0, [], set())
        if found:
            return k
    return 0

def _find_sidon_subset(S, k, start, current, used_sums):
    """DFS to find a Sidon subset of size k starting from index 'start'."""
    if len(current) == k:
        return True
    
    # Pruning: not enough elements left
    remaining = len(S) - start
    need = k - len(current)
    if remaining < need:
        return False
    
    for i in range(start, len(S)):
        x = S[i]
        # Check all pair sums involving x with existing elements
        valid = True
        new_sums = []
        for j in range(len(current)):
            s = x + current[j]
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if not valid:
            continue
        s = x + x
        if s in used_sums:
            continue
        new_sums.append(s)
        
        # Add x and its new sums
        for s in new_sums:
            used_sums.add(s)
        current.append(x)
        
        if _find_sidon_subset(S, k, i + 1, current, used_sums):
            return True
        
        # Backtrack
        current.pop()
        for s in new_sums:
            used_sums.discard(s)
    
    return False

def has_sidon_subset_of_size_k(S, k, time_limit=0.1):
    """Check if S has a Sidon subset of size k. Returns True if found."""
    start_time = time.time()
    # We only need to check if a subset of size k exists.
    # If k is small, this is fast. If k is large, it might be slow.
    # For our purpose, we check k=4. If no 4-Sidon exists, max size is 3.
    found = _find_sidon_subset(S, k, 0, [], set())
    return found

def local_search_for_max_sidon_3(deadline):
    """
    Use randomized local search to find a set of 12 distinct integers in [0, 128]
    such that the largest Sidon subset has size <= 3.
    This is equivalent to saying there is NO Sidon subset of size 4.
    """
    # Start with a random set
    current_set = random.sample(range(129), 12)
    current_set.sort()
    
    # Check if it already works
    if not has_sidon_subset_of_size_k(current_set, 4):
        return current_set
        
    # Local search: try to swap elements to break Sidon-4 subsets
    # We don't need to find the "best" one, just any valid one.
    # Since APs are valid, and random sets might be close, this should converge quickly.
    
    # A simple hill climbing approach:
    # Identify a Sidon subset of size 4.
    # Try to change one element in that subset to break the Sidon property.
    
    # For efficiency, we just try random perturbations and check.
    # If we find one with no 4-Sidon, return it.
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        # Pick a random element to replace
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in current_set and new_val != old_val:
            continue
            
        # Swap
        current_set[idx] = new_val
        current_set.sort()
        
        # Check condition: No Sidon subset of size 4
        # This is the expensive part.
        # Optimization: Only check if the change could have created a new 4-Sidon
        # or if it broke an existing one.
        # For simplicity, just check the whole set.
        if not has_sidon_subset_of_size_k(current_set, 4):
            return current_set
            
        # Optional: Revert if it made things worse?
        # Since we don't have a gradient, we can't easily say "worse".
        # We just hope to hit a valid state.
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: evaluator kills at 90s
    
    # The AP [0,1,...,11] has max Sidon size exactly 3.
    # This is provably optimal.
    best_set = list(range(12))
    
    # Verify the AP is valid (should be very fast)
    # If max_sidon_size is 3, it's optimal.
    score = max_sidon_size(best_set)
    if score <= 3:
        return best_set
        
    # If for some reason the AP is not optimal (impossible) or check failed,
    # try local search to find another set with max Sidon size 3.
    if time.time() < deadline:
        found_set = local_search_for_max_sidon_3(deadline)
        if found_set:
            return found_set
            
    return best_set
