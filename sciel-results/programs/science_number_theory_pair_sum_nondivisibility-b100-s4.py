"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50
    n = 256
    
    # Precompute conflicts using bitmasks for speed
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # Count conflicts for each number
    conflict_count = [0] * (n + 1)
    for a in range(1, n + 1):
        mask = conflict_mask[a]
        while mask:
            conflict_count[a] += 1
            mask &= mask - 1
    
    best = list(range(1, n + 1, 2))
    best_size = len(best)
    
    def is_compatible(x, current_mask):
        """Check if x can be added to set represented by current_mask"""
        return (conflict_mask[x] & current_mask) == 0
    
    def greedy_add_sorted(order, start_mask):
        """Greedy add elements in given order to start_mask"""
        current_mask = start_mask
        for e in order:
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
        return current_mask
    
    def count_bits(mask):
        """Count set bits in mask"""
        c = 0
        while mask:
            c += 1
            mask &= mask - 1
        return c
    
    # Convert best to mask
    best_mask = 0
    for x in best:
        best_mask |= (1 << x)
    
    # Sort evens by conflict count (add low-conflict ones first)
    evens = list(range(2, n + 1, 2))
    evens_sorted = sorted(evens, key=lambda x: conflict_count[x])
    
    # Try greedy with different orderings
    odds_mask = 0
    for x in range(1, n + 1, 2):
        odds_mask |= (1 << x)
    
    # First: sorted by conflict count
    result_mask = greedy_add_sorted(evens_sorted, odds_mask)
    if count_bits(result_mask) > best_size:
        best_size = count_bits(result_mask)
        best_mask = result_mask
    
    # Try random permutations multiple times
    iter_count = 0
    while time.time() < deadline - 15:
        order = evens[:]
        random.shuffle(order)
        result_mask = greedy_add_sorted(order, odds_mask)
        if count_bits(result_mask) > best_size:
            best_size = count_bits(result_mask)
            best_mask = result_mask
        iter_count += 1
        
        # Periodic local search on best
        if iter_count % 100 == 0:
            # Try to add any compatible element
            improved = True
            while improved:
                improved = False
                for x in range(1, n + 1):
                    if not (best_mask >> x) & 1:
                        if (conflict_mask[x] & best_mask) == 0:
                            best_mask |= (1 << x)
                            best_size += 1
                            improved = True
                            break
    
    # Also try starting from different bases
    # Try all numbers sorted by conflict count
    all_nums = list(range(1, n + 1))
    all_sorted = sorted(all_nums, key=lambda x: conflict_count[x])
    
    # Greedy from empty set
    empty_mask = 0
    result_mask = greedy_add_sorted(all_sorted, empty_mask)
    if count_bits(result_mask) > best_size:
        best_size = count_bits(result_mask)
        best_mask = result_mask
    
    # Try random orders from empty set
    while time.time() < deadline - 5:
        order = all_nums[:]
        random.shuffle(order)
        result_mask = greedy_add_sorted(order, empty_mask)
        if count_bits(result_mask) > best_size:
            best_size = count_bits(result_mask)
            best_mask = result_mask
        
        # Local search
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask >> x) & 1:
                    if (conflict_mask[x] & best_mask) == 0:
                        best_mask |= (1 << x)
                        best_size += 1
                        improved = True
                        break
    
    # Convert best_mask back to list
    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return best
# EVOLVE-BLOCK-END
