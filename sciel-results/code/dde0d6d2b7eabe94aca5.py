"""Program that searches for 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _all_subset_sums_distinct(seq):
    """Check if all subset sums are distinct using a bitset. Returns True if valid."""
    n = len(seq)
    bits = 1  # only sum 0 initially
    for a in seq:
        bits |= (bits << a)
    # Count set bits; must equal 2^n
    return bin(bits).count('1') == (1 << n)

def _count_subset_sums(seq):
    """Count distinct subset sums using bitset."""
    bits = 1
    for a in seq:
        bits |= (bits << a)
    return bin(bits).count('1')

def _greedy_construct(n):
    """Build the minimal superincreasing (binary) sequence: 1, 2, 4, ..., 2^(n-1)."""
    return [1 << i for i in range(n)]

def _local_search(a, time_limit, random_seed=None):
    """
    Local search to reduce the largest element.
    Uses bitset-based validity checking for speed.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    a = sorted(a)
    best = a[:]
    best_max = max(a)
    start_time = time.time()
    n = len(a)
    
    # Try to reduce elements, starting from the largest
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit - 2:
            break
        
        for idx in range(n - 1, -1, -1):
            if time.time() - start_time > time_limit - 2:
                break
            
            old_val = best[idx]
            # Try reducing this element
            for new_val in range(1, old_val):
                if time.time() - start_time > time_limit - 2:
                    break
                
                candidate = best[:]
                candidate[idx] = new_val
                candidate.sort()
                
                # Check distinctness
                if len(set(candidate)) != n:
                    continue
                
                # Check validity
                if _all_subset_sums_distinct(candidate):
                    new_max = max(candidate)
                    if new_max < best_max:
                        best = candidate
                        best_max = new_max
                        improved = True
                        break
                    elif new_max == best_max and sum(candidate) < sum(best):
                        best = candidate
                        improved = True
                        break
            if improved:
                break
    
    return best

def _random_restart_search(n, max_val, time_limit):
    """
    Random restart local search to find a better solution.
    """
    start_time = time.time()
    best_seq = _greedy_construct(n)
    best_max = max(best_seq)
    
    rng = random.Random(42)
    
    # Try local search from the greedy solution first
    improved = _local_search(best_seq, min(10.0, time_limit - 5), seed=42)
    if max(improved) < best_max:
        best_seq = improved
        best_max = max(best_seq)
    
    # Random restarts
    while time.time() - start_time < time_limit - 5:
        # Perturb the best sequence
        base = best_seq[:]
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, n - 1)
            offset = rng.randint(-100, 100)
            new_val = base[idx] + offset
            if 1 <= new_val <= max_val:
                base[idx] = new_val
        
        base.sort()
        
        if len(set(base)) != n:
            continue
        
        if _all_subset_sums_distinct(base):
            seq = _local_search(base, min(5.0, time_limit - (time.time() - start_time) - 5), seed=rng.randint(0, 1000000))
            if max(seq) < best_max:
                best_seq = seq
                best_max = max(seq)
        
        if best_max <= 2000:
            break
    
    return best_seq

def construct():
    """
    Construct 12 distinct positive integers with distinct subset sums, minimizing the largest element.
    """
    start_time = time.time()
    n = 12
    max_val = 1048576
    time_limit = 60.0  # Safety margin before 90s kill
    
    # Start with the binary sequence
    best_seq = _greedy_construct(n)
    best_max = max(best_seq)
    
    # Try to improve with local search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        improved = _local_search(best_seq, min(15.0, remaining - 5), seed=42)
        if max(improved) < best_max:
            best_seq = improved
            best_max = max(best_seq)
    
    # Try random restarts if time permits
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        seq = _random_restart_search(n, max_val, remaining)
        if max(seq) < best_max:
            best_seq = seq
            best_max = max(seq)
    
    return best_seq

# EVOLVE-BLOCK-END
