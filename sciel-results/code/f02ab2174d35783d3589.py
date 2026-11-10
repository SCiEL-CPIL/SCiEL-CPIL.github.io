# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences F = {x^2 mod 205 : x in 1..204} \ {0}.
    2. Precompute conflict bitmasks for O(1) conflict checking.
    3. Use greedy initialization with many random orderings,
       followed by simulated annealing with add/remove/swap moves using bitmask checks,
       and final greedy fill to maximize set size.
    4. Include a deadline guard to ensure we finish well before the 600s limit.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 570  # Safety margin of 30s
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    forbidden_set = forbidden  # for fast lookup
    
    # Precompute conflict bitmasks: conflict_mask[i] is a bitmask of all j that conflict with i
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden_set:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Also build a set-based conflict matrix for verification
    conflict_set = [set() for _ in range(m)]
    for i in range(m):
        mask = conflict_mask[i]
        for j in range(m):
            if (mask >> j) & 1:
                conflict_set[i].add(j)
    
    def can_add_bitmask(cand, sel_mask):
        """Check if cand can be added to set represented by sel_mask without conflicts. O(1) bit op."""
        return (conflict_mask[cand] & sel_mask) == 0
    
    def greedy_fill_bitmask(sel_mask, order=None):
        """Greedy fill: add elements in order (or 0..m-1) if they don't conflict."""
        if order is None:
            order = range(m)
        for cand in order:
            if not (sel_mask >> cand) & 1:
                if can_add_bitmask(cand, sel_mask):
                    sel_mask |= (1 << cand)
        return sel_mask
    
    def mask_to_list(sel_mask):
        """Convert bitmask to list of set bits."""
        result = []
        for i in range(m):
            if (sel_mask >> i) & 1:
                result.append(i)
        return result
    
    def list_to_mask(lst):
        """Convert list to bitmask."""
        mask = 0
        for x in lst:
            mask |= (1 << x)
        return mask
    
    # Greedy initialization: try multiple orderings and take the best
    best_init_mask = 0
    best_init_size = -1
    
    # Deterministic orderings
    det_orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        list(range(0, m, 2)),
        list(range(1, m, 2)),
        list(range(0, m, 3)),
        list(range(1, m, 3)),
        list(range(2, m, 3)),
        list(range(0, m, 5)),
        list(range(1, m, 5)),
        list(range(2, m, 5)),
        list(range(3, m, 5)),
        list(range(4, m, 5)),
        list(range(0, m, 7)),
        list(range(1, m, 7)),
        list(range(2, m, 7)),
        list(range(0, m, 11)),
        list(range(1, m, 11)),
        list(range(0, m, 13)),
        list(range(1, m, 13)),
        list(range(0, m, 17)),
        list(range(1, m, 17)),
        list(range(0, m, 19)),
        list(range(1, m, 19)),
        list(range(0, m, 23)),
        list(range(1, m, 23)),
        list(range(0, m, 29)),
        list(range(1, m, 29)),
    ]
    
    for order in det_orders:
        sel_mask = greedy_fill_bitmask(0, order)
        sz = bin(sel_mask).count('1')
        if sz > best_init_size:
            best_init_size = sz
            best_init_mask = sel_mask
    
    # Many random orderings for better initialization
    for seed in range(200):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        sel_mask = greedy_fill_bitmask(0, order)
        sz = bin(sel_mask).count('1')
        if sz > best_init_size:
            best_init_size = sz
            best_init_mask = sel_mask
    
    if best_init_mask == 0:
        best_init_mask = 1  # just element 0
        best_init_size = 1
    
    # Start with the best greedy solution
    current_mask = best_init_mask
    current_size = best_init_size
    
    # Track best found so far
    best_mask = current_mask
    best_size = current_size
    
    # Phase 1: Greedy fill on current set
    current_mask = greedy_fill_bitmask(current_mask)
    current_size = bin(current_mask).count('1')
    if current_size > best_size:
        best_size = current_size
        best_mask = current_mask
    
    # Phase 2: Simulated annealing with bitmask-based conflict checks
    rng = random.Random(98765)
    temp = 2.0
    cooling = 0.9998
    min_temp = 0.01
    
    # Precompute list of all elements for random selection
    all_elements = list(range(m))
    
    iteration = 0
    while time.time() < deadline - 20 and temp > min_temp:
        iteration += 1
        
        if iteration % 20000 == 0:
            temp = max(temp * cooling, min_temp)
        
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.35:
            # Try to add a random element not in set
            # Find candidates not in set
            not_in_set = [c for c in all_elements if not (current_mask >> c) & 1]
            if not not_in_set:
                continue
            cand = rng.choice(not_in_set)
            if can_add_bitmask(cand, current_mask):
                new_mask = current_mask | (1 << cand)
                current_mask = new_mask
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
        
        elif move_type < 0.70:
            # Swap: remove one from set, add one from outside
            in_set = [c for c in all_elements if (current_mask >> c) & 1]
            if not in_set:
                continue
            to_remove = rng.choice(in_set)
            not_in_set = [c for c in all_elements if not (current_mask >> c) & 1]
            if not not_in_set:
                continue
            to_add = rng.choice(not_in_set)
            
            # Check if to_add conflicts with (current set minus to_remove)
            reduced_mask = current_mask & ~(1 << to_remove)
            if can_add_bitmask(to_add, reduced_mask):
                new_mask = (current_mask & ~(1 << to_remove)) | (1 << to_add)
                current_mask = new_mask
                # Size stays the same
        
        elif move_type < 0.85:
            # Remove one (to break out of local optima)
            in_set = [c for c in all_elements if (current_mask >> c) & 1]
            if not in_set:
                continue
            to_remove = rng.choice(in_set)
            current_mask &= ~(1 << to_remove)
            current_size -= 1
        
        else:
            # Double swap: remove two, add two (if possible)
            in_set = [c for c in all_elements if (current_mask >> c) & 1]
            not_in_set = [c for c in all_elements if not (current_mask >> c) & 1]
            if len(in_set) < 2 or len(not_in_set) < 2:
                continue
            # Pick two to remove
            idx1 = rng.randrange(len(in_set))
            idx2 = rng.randrange(len(in_set) - 1)
            if idx2 >= idx1:
                idx2 += 1
            r1 = in_set[idx1]
            r2 = in_set[idx2]
            # Pick two to add
            idx3 = rng.randrange(len(not_in_set))
            idx4 = rng.randrange(len(not_in_set) - 1)
            if idx4 >= idx3:
                idx4 += 1
            a1 = not_in_set[idx3]
            a2 = not_in_set[idx4]
            
            # Check if a1 and a2 can be added to (current set minus r1, r2)
            reduced_mask = current_mask & ~(1 << r1) & ~(1 << r2)
            if can_add_bitmask(a1, reduced_mask) and can_add_bitmask(a2, reduced_mask | (1 << a1)):
                new_mask = (current_mask & ~(1 << r1) & ~(1 << r2)) | (1 << a1) | (1 << a2)
                current_mask = new_mask
                # Size stays the same
        
        # Cool down temperature
        temp *= cooling
    
    # Phase 3: Final greedy fill on the best set found
    final_mask = greedy_fill_bitmask(best_mask)
    final_size = bin(final_mask).count('1')
    
    # Also try greedy fill on current set
    curr_filled = greedy_fill_bitmask(current_mask)
    curr_filled_size = bin(curr_filled).count('1')
    if curr_filled_size > final_size:
        final_mask = curr_filled
        final_size = curr_filled_size
    
    # Additional: try multiple random greedy fills starting from best
    for seed in range(50):
        rng2 = random.Random(10000 + seed)
        order = list(range(m))
        rng2.shuffle(order)
        # Start from best_mask and greedily fill in random order
        trial_mask = best_mask
        for cand in order:
            if not (trial_mask >> cand) & 1:
                if can_add_bitmask(cand, trial_mask):
                    trial_mask |= (1 << cand)
        trial_size = bin(trial_mask).count('1')
        if trial_size > final_size:
            final_mask = trial_mask
            final_size = trial_size
    
    # Convert final mask to list
    final_list = mask_to_list(final_mask)
    
    # Verify the final set
    for i in range(len(final_list)):
        for j in range(i+1, len(final_list)):
            a, b = final_list[i], final_list[j]
            if b in conflict_set[a]:
                # Should not happen, but if it does, rebuild
                final_list = [0]
                break
    
    return np.array(final_list, dtype=np.int64)

# EVOLVE-BLOCK-END
