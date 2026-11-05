"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    weights = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflict masks using bit operations
    # conflict_mask[x] has bit y set if x and y are in some 3-term AP
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            # a, b, c form 3-term AP
            conflict_mask[a] |= (1 << (b - 1)) | (1 << (c - 1))
            conflict_mask[b] |= (1 << (a - 1)) | (1 << (c - 1))
            conflict_mask[c] |= (1 << (a - 1)) | (1 << (b - 1))
    
    def is_valid_addition(selected_mask, x):
        """O(1) check if x can be added."""
        if (selected_mask >> (x - 1)) & 1:
            return False
        return (conflict_mask[x] & selected_mask) == 0
    
    def reciprocal_sum_from_mask(mask):
        """Compute reciprocal sum."""
        total = 0.0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                total += weights[x]
        return total
    
    def greedy_from_mask(mask):
        """Greedily add valid elements, smallest x first."""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not ((mask >> (x - 1)) & 1):
                    if is_valid_addition(mask, x):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
        return mask
    
    def local_search_fast(mask):
        """Local search: try removing each element, greedily refill, accept if better."""
        current_weight = reciprocal_sum_from_mask(mask)
        improved = True
        while improved:
            improved = False
            # Try removing from largest to smallest (cheapest to remove first)
            for x in range(N, 0, -1):
                if not ((mask >> (x - 1)) & 1):
                    continue
                temp_mask = mask & ~(1 << (x - 1))
                new_mask = greedy_from_mask(temp_mask)
                new_weight = reciprocal_sum_from_mask(new_mask)
                if new_weight > current_weight + 1e-12:
                    mask = new_mask
                    current_weight = new_weight
                    improved = True
                    break
        return mask
    
    def pair_swap_search(mask):
        """Try swapping: remove one element, add another."""
        current_weight = reciprocal_sum_from_mask(mask)
        improved = True
        while improved:
            improved = False
            for x in range(N, 0, -1):
                if not ((mask >> (x - 1)) & 1):
                    continue
                temp_mask = mask & ~(1 << (x - 1))
                # Try adding each y not in temp_mask
                for y in range(1, N + 1):
                    if (temp_mask >> (y - 1)) & 1:
                        continue
                    if is_valid_addition(temp_mask, y):
                        new_mask = temp_mask | (1 << (y - 1))
                        new_weight = current_weight - weights[x] + weights[y]
                        if new_weight > current_weight + 1e-12:
                            mask = new_mask
                            current_weight = new_weight
                            improved = True
                            break
                if improved:
                    break
        return mask
    
    # Phase 1: Multiple greedy initializations
    best_mask = 0
    best_weight = 0.0
    
    orders = [list(range(1, N + 1)), list(range(N, 0, -1))]
    for _ in range(30):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        bm = 0
        for x in order:
            if is_valid_addition(bm, x):
                bm |= (1 << (x - 1))
        bm = greedy_from_mask(bm)
        w = reciprocal_sum_from_mask(bm)
        if w > best_weight:
            best_weight = w
            best_mask = bm
    
    # Phase 2: Local search
    best_mask = local_search_fast(best_mask)
    best_weight = reciprocal_sum_from_mask(best_mask)
    
    # Phase 3: Pair swap search
    best_mask = pair_swap_search(best_mask)
    best_weight = reciprocal_sum_from_mask(best_mask)
    
    # Phase 4: Multiple restarts with perturbation
    while (time.time() - start_time) < DEADLINE - 5:
        current = best_mask
        elems = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
        if not elems:
            break
        
        num_remove = random.randint(1, min(10, len(elems)))
        to_remove = random.sample(elems, num_remove)
        for r in to_remove:
            current &= ~(1 << (r - 1))
        
        current = greedy_from_mask(current)
        current = local_search_fast(current)
        current = pair_swap_search(current)
        w = reciprocal_sum_from_mask(current)
        
        if w > best_weight + 1e-12:
            best_weight = w
            best_mask = current
    
    # Final polish
    best_mask = local_search_fast(best_mask)
    best_mask = pair_swap_search(best_mask)
    best_weight = reciprocal_sum_from_mask(best_mask)
    
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return result
# EVOLVE-BLOCK-END
