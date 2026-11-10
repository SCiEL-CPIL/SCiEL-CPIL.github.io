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
    
    def local_search_1add(mask, size):
        """Try to add any compatible element repeatedly"""
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if not (mask >> x) & 1:
                    if (conflict_mask[x] & mask) == 0:
                        mask |= (1 << x)
                        size += 1
                        improved = True
                        break
        return mask, size
    
    def local_search_2swap(mask, size):
        """Try 2-swap: remove two elements, try to add three compatible ones"""
        best_mask = mask
        best_size = size
        elems = [x for x in range(1, n + 1) if (mask >> x) & 1]
        
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                # Remove elems[i] and elems[j]
                temp_mask = mask
                temp_mask &= ~(1 << elems[i])
                temp_mask &= ~(1 << elems[j])
                temp_size = size - 2
                
                # Try to add compatible elements
                added = 0
                for x in range(1, n + 1):
                    if not (temp_mask >> x) & 1:
                        if (conflict_mask[x] & temp_mask) == 0:
                            temp_mask |= (1 << x)
                            temp_size += 1
                            added += 1
                            if temp_size > best_size:
                                best_mask = temp_mask
                                best_size = temp_size
        
        return best_mask, best_size
    
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
    
    # Try random permutations multiple times with noise-perturbed ordering
    iter_count = 0
    while time.time() < deadline - 15:
        # Create noise-perturbed ordering
        order = evens[:]
        random.shuffle(order)
        # Sort by noisy conflict count
        noisy_order = sorted(order, key=lambda x: conflict_count[x] + random.random() * 5)
        result_mask = greedy_add_sorted(noisy_order, odds_mask)
        if count_bits(result_mask) > best_size:
            best_size = count_bits(result_mask)
            best_mask = result_mask
        iter_count += 1
        
        # Periodic local search on best
        if iter_count % 50 == 0:
            best_mask, best_size = local_search_1add(best_mask, best_size)
            # Try 2-swap less frequently (more expensive)
            if iter_count % 200 == 0:
                best_mask, best_size = local_search_2swap(best_mask, best_size)
    
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
    
    # Try random orders from empty set with noise
    while time.time() < deadline - 5:
        order = all_nums[:]
        random.shuffle(order)
        noisy_order = sorted(order, key=lambda x: conflict_count[x] + random.random() * 5)
        result_mask = greedy_add_sorted(noisy_order, empty_mask)
        if count_bits(result_mask) > best_size:
            best_size = count_bits(result_mask)
            best_mask = result_mask
        
        # Local search
        best_mask, best_size = local_search_1add(best_mask, best_size)
        # Occasionally try 2-swap
        if random.random() < 0.1:
            best_mask, best_size = local_search_2swap(best_mask, best_size)
    
    # Final polish: ensure no single additions are possible
    best_mask, best_size = local_search_1add(best_mask, best_size)
    
    # Convert best_mask back to list
    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return best
# EVOLVE-BLOCK-END
