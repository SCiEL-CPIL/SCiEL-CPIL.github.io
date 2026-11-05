"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    # A sum that appears k times contributes C(k,2) duplicate pairs
    return sum(k * (k - 1) // 2 for k in c.values())

def _max_sidon_subset_size_bruteforce(subset, deadline=None):
    """Find largest subset with all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    best = 0
    # Try all subsets in descending size order
    for size in range(n, best, -1):
        if deadline and time.time() > deadline:
            break
        # Generate all combinations of this size
        from itertools import combinations
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                break
            # Check if this subset is Sidon (all pair sums unique)
            vals = [subset[i] for i in combo]
            s = len(vals)
            pair_sums = set()
            is_sidon = True
            for i in range(s):
                for j in range(i, s):
                    ps = vals[i] + vals[j]
                    if ps in pair_sums:
                        is_sidon = False
                        break
                    pair_sums.add(ps)
                if not is_sidon:
                    break
            if is_sidon:
                best = size
                break
        if best == size:
            break
    return best

def _greedy_collision_swap(subset, deadline, max_swaps=5000):
    """Greedily swap elements to maximize duplicate pair sums."""
    subset = list(subset)
    n = len(subset)
    best_dups = _count_duplicate_sums(subset)
    
    for k in range(max_swaps):
        # Check time periodically to avoid overshooting
        if k % 100 == 0 and time.time() > deadline:
            break
            
        # Try replacing element i with a random value in [0, 128]
        i = random.randint(0, n - 1)
        old_val = subset[i]
        new_val = random.randint(0, 128)
        # Check for duplicates efficiently
        # Since n=12, `new_val in subset` is O(12) which is fine.
        # But we can do better by checking `subset.count(new_val) == 0` or similar.
        # Actually, `new_val in subset` creates a generator or scans list.
        # For small n, it's fine.
        if new_val in subset and new_val != old_val:
            continue
            
        subset[i] = new_val
        new_dups = _count_duplicate_sums(subset)
        if new_dups > best_dups:
            best_dups = new_dups
        else:
            subset[i] = old_val
    
    return subset, best_dups

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Initial candidate: try to create maximum collisions
    # Start with a structured set that has many duplicate sums
    # e.g., pairs that sum to the same value
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_bruteforce(best_subset, deadline)
    
    # Generate random candidates and improve via greedy swap
    random.seed(42)
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Check time again to avoid overshooting
        if time.time() > deadline - 5:
            break
            
        # Generate a random 12-element subset
        candidate = random.sample(range(129), 12)
        
        # Improve via greedy collision maximization
        # Reduce max_swaps if time is running out
        time_left = deadline - time.time()
        if time_left < 10:
            max_swaps = 500
        elif time_left < 30:
            max_swaps = 1000
        else:
            max_swaps = 2000
            
        improved, dups = _greedy_collision_swap(candidate, deadline, max_swaps=max_swaps)
        
        # Check time before expensive brute force
        if time.time() > deadline - 2:
            break
            
        # Evaluate
        score = _max_sidon_subset_size_bruteforce(improved, deadline)
        if score < best_score:
            best_score = score
            best_subset = improved
            if best_score <= 4:  # Theoretical minimum for 12 elements is around 4-5
                break
    
    return sorted(best_subset)
# EVOLVE-BLOCK-END
