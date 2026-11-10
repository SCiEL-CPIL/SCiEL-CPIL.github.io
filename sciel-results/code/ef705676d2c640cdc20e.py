"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # Elements are 2..256, index 0 = element 2, index i = element i+2
    # n elements = 255
    n = N - 1  # 255 elements
    
    # Precompute conflicts: conflict[i][j] = True if lcm(el_i, el_j) <= 256
    # Use bitmasks for fast conflict checking
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    conflict_mask = [0] * n
    elements = list(range(2, 257))
    
    for i in range(n):
        ai = elements[i]
        for j in range(i + 1, n):
            aj = elements[j]
            g = gcd(ai, aj)
            if (ai * aj) // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Reciprocal weights
    weights = [1.0 / elements[i] for i in range(n)]
    
    def total_weight(selected_mask):
        """Sum of reciprocals for selected set (bitmask)."""
        s = 0.0
        m = selected_mask
        while m:
            i = (m & -m).bit_length() - 1
            s += weights[i]
            m &= m - 1
        return s
    
    def is_valid(selected_mask):
        """Check if no two selected elements conflict."""
        m = selected_mask
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            # Check if any other selected element conflicts with i
            if (conflict_mask[i] & m) != 0:
                return False
        return True
    
    def can_add(selected_mask, i):
        """Check if element i can be added to selected set."""
        return (conflict_mask[i] & selected_mask) == 0
    
    # Greedy construction with different orderings
    def greedy_order(order):
        """Build a valid set by adding elements in given order."""
        selected = 0
        for i in order:
            if can_add(selected, i):
                selected |= (1 << i)
        return selected
    
    # Try multiple greedy strategies
    best_mask = 0
    best_val = 0.0
    
    # Strategy 1: Greedy by descending weight (ascending element)
    order1 = list(range(n))  # elements 2,3,4,...256 -> indices 0,1,2,...
    mask1 = greedy_order(order1)
    val1 = total_weight(mask1)
    if val1 > best_val:
        best_val = val1
        best_mask = mask1
    
    # Strategy 2: Greedy by descending weight but skip some small ones
    # Try: start from large elements first (lower reciprocal but less conflict)
    order2 = list(range(n - 1, -1, -1))  # 256, 255, ..., 2
    mask2 = greedy_order(order2)
    val2 = total_weight(mask2)
    if val2 > best_val:
        best_val = val2
        best_mask = mask2
    
    # Strategy 3: Randomized greedy
    for _ in range(20):
        if time.time() - start_time > DEADLINE:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        mask_r = greedy_order(random_order)
        val_r = total_weight(mask_r)
        if val_r > best_val:
            best_val = val_r
            best_mask = mask_r
    
    # Local search: 1-for-1 and 2-for-1 exchanges
    def local_search(mask, max_iter=50000):
        """Improve mask via 1-for-1 and 2-for-1 exchanges."""
        current = mask
        current_val = total_weight(current)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() - start_time > DEADLINE:
                break
            
            # Get list of selected elements
            selected = []
            m = current
            while m:
                i = (m & -m).bit_length() - 1
                selected.append(i)
                m &= m - 1
            
            # 1-for-1: try removing one selected and adding one unselected
            for i in selected:
                if time.time() - start_time > DEADLINE:
                    break
                new_mask = current & ~(1 << i)
                # Try adding each unselected element
                for j in range(n):
                    if (current >> j) & 1:
                        continue
                    if can_add(new_mask, j):
                        candidate = new_mask | (1 << j)
                        delta = weights[j] - weights[i]
                        if delta > 1e-15:
                            current = candidate
                            current_val += delta
                            improved = True
                            break
            
            if improved:
                continue
            
            # 2-for-1: remove two selected, add one unselected
            for idx_a in range(len(selected)):
                if time.time() - start_time > DEADLINE:
                    break
                for idx_b in range(idx_a + 1, len(selected)):
                    i = selected[idx_a]
                    j = selected[idx_b]
                    new_mask = current & ~(1 << i) & ~(1 << j)
                    for k in range(n):
                        if (current >> k) & 1:
                            continue
                        if can_add(new_mask, k):
                            candidate = new_mask | (1 << k)
                            delta = weights[k] - weights[i] - weights[j]
                            if delta > 1e-15:
                                current = candidate
                                current_val += delta
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return current, current_val
    
    # Run local search from best greedy solution
    best_mask, best_val = local_search(best_mask)
    
    # Try local search from other starting points
    for _ in range(10):
        if time.time() - start_time > DEADLINE:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        mask_r = greedy_order(random_order)
        val_r = total_weight(mask_r)
        if val_r > best_val * 0.95:  # Only if close
            mask_ls, val_ls = local_search(mask_r, max_iter=20000)
            if val_ls > best_val:
                best_val = val_ls
                best_mask = mask_ls
    
    # Final: try to add any remaining elements
    for i in range(n):
        if time.time() - start_time > DEADLINE:
            break
        if not (best_mask >> i) & 1:
            if can_add(best_mask, i):
                best_mask |= (1 << i)
                best_val += weights[i]
    
    # Convert bitmask back to list
    result = []
    m = best_mask
    while m:
        i = (m & -m).bit_length() - 1
        result.append(elements[i])
        m &= m - 1
    
    return result
# EVOLVE-BLOCK-END
