import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return a & b == a & c == b & c

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Pre-compute all 4-element subsets of [1,16] as bitmasks for speed
    # and also keep the original tuples for output
    all_sets_info = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for i in row:
            mask |= (1 << (i-1))
        all_sets_info.append((mask, row))
    
    n_total = len(all_sets_info)
    
    # Helper to run greedy given an order of indices
    def greedy_pass(order):
        chosen_masks = []
        chosen_rows = []
        for idx in order:
            mask, row = all_sets_info[idx]
            s = set(row)
            can_add = True
            # Check against all pairs in chosen
            # Optimization: If chosen is large, this is O(K^2).
            # K is max 250. Pairs ~ 30k. 1820 * 30k = 54M checks.
            # This might be too slow for Python in 85s if we do many passes.
            # 54M set ops in Python is likely ~5-10 seconds per pass.
            # So we can do ~5-10 passes in 85s.
            
            # To speed up, we can use the bitmask representation for intersection check?
            # a & b == a & c == b & c
            # Using sets is clearer. Let's stick to sets but optimize the loop.
            
            # Actually, let's use the bitmask for the check too?
            # No, the sunflower condition is about the *sets* of elements.
            # Intersection equality is equivalent to bitmask equality.
            # So: (mask_a & mask_b) == (mask_a & mask_c) == (mask_b & mask_c)
            
            if can_add:
                for i in range(len(chosen_masks)):
                    if not can_add:
                        break
                    ma = chosen_masks[i]
                    for j in range(i + 1, len(chosen_masks)):
                        mb = chosen_masks[j]
                        if (ma & mb) == (ma & mask) == (mb & mask):
                            can_add = False
                            break
            
            if can_add:
                chosen_masks.append(mask)
                chosen_rows.append(row)
                
        return chosen_rows

    best_family = []
    best_size = 0
    
    # Define several deterministic orderings
    indices = list(range(n_total))
    
    # 1. Lexicographic (default)
    order_1 = indices[:]
    
    # 2. Reverse Lexicographic
    order_2 = indices[::-1]
    
    # 3. Sorted by bitmask value
    order_3 = sorted(indices, key=lambda i: all_sets_info[i][0])
    
    # 4. Sorted by sum of elements
    order_4 = sorted(indices, key=lambda i: sum(all_sets_info[i][1]))
    
    # 5. Sorted by "first element" then "last element" ?
    # Just use a few simple ones.
    
    orders = [order_1, order_2, order_3, order_4]
    
    # Add a couple of random orders just in case, but prioritize deterministic
    # Since random scored worse in current program, we put them last or limit them.
    for _ in range(2):
        order_r = indices[:]
        random.shuffle(order_r)
        orders.append(order_r)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        family = greedy_pass(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # If we hit the cap, we can stop
        if best_size >= 250:
            break
            
    return [sorted(s) for s in best_family]
