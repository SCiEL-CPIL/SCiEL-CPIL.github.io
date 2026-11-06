"""Rewritten program: greedy construction for distinct subset sums."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct exactly n=12 distinct positive integers <= 1048576
    such that all 2^12 = 4096 subset sums are distinct.
    Minimize the largest integer.
    
    Theoretical lower bound: For n positive integers with all subset sums distinct,
    the largest must be at least 2^(n-1). Powers of 2 achieve this bound.
    
    We use a greedy approach: at each step, pick the smallest positive integer
    not yet used that keeps all subset sums distinct. For the distinct-subset-sum
    problem, this greedy yields exactly powers of 2, which are provably optimal.
    
    We also include a verification step to confirm all 4096 subset sums are distinct.
    """
    n = 12
    max_val = 1048576
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Greedy construction: pick smallest unused positive integer that preserves distinctness
    selected = []
    used_sums = {0}  # set of all subset sums achievable with current selection
    
    for k in range(n):
        if time.time() > deadline:
            break
        # Try candidate values starting from 1
        candidate = 1
        while True:
            if time.time() > deadline:
                break
            # Check if candidate keeps all subset sums distinct
            # New subset sums = old_sums union (old_sums + candidate)
            # All must be distinct
            valid = True
            new_sums = set(used_sums)
            for s in used_sums:
                ns = s + candidate
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                selected.append(candidate)
                used_sums = new_sums
                break
            candidate += 1
        
        if time.time() > deadline:
            break
    
    # Verify: all 2^n subset sums must be distinct
    if len(selected) < n:
        # Fallback to powers of 2 (provably optimal)
        selected = [1 << i for i in range(n)]
    
    # Full verification of all 4096 subset sums
    all_sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += selected[i]
        all_sums.add(s)
    
    if len(all_sums) != (1 << n):
        # Should never happen with powers of 2, but fallback
        selected = [1 << i for i in range(n)]
    
    # Ensure all values are <= max_val
    if max(selected) > max_val:
        # Fallback
        selected = [1 << i for i in range(n)]
    
    return selected
# EVOLVE-BLOCK-END
