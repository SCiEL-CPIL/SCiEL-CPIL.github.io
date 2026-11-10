# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses bitmask-based local search with 1-swap, 2-swap, and add/remove moves.
    """
    m = 205
    deadline = time.time() + 615  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)
    
    # Build adjacency masks: adj_mask[i] has bit j set iff (i-j) mod m is in forbidden
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden_list:
            j = (i + f) % m
            mask |= (1 << j)
        adj_mask[i] = mask

    # all_mask: all 205 bits set
    all_mask = (1 << m) - 1

    def can_add(candidate, current_mask):
        """Check if candidate can be added to the set represented by current_mask."""
        # candidate must not be in current_mask
        if current_mask & (1 << candidate):
            return False
        # candidate must not conflict with any element in current_mask
        # i.e., adj_mask[candidate] & current_mask must be 0
        return (adj_mask[candidate] & current_mask) == 0

    def greedy_init(order, rng=None):
        """Greedy construction following given order."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            if can_add(candidate, selected_mask):
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def local_search(S_list, S_mask, max_iters=20000):
        """
        Local search with add, 1-swap, and 2-swap moves using bitmask representation.
        Returns (best_list, best_mask).
        """
        best_list = list(S_list)
        best_mask = S_mask
        best_size = len(best_list)
        
        # Precompute for each element the set of elements it conflicts with
        # This is already in adj_mask
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try add-only moves first (cheapest)
            added = False
            for a in range(m):
                if can_add(a, best_mask):
                    best_mask |= (1 << a)
                    best_list.append(a)
                    added = True
                    improved = True
                    break
            
            if added:
                best_size = len(best_list)
                continue
            
            # Try 1-swap moves: remove r, add a
            found_swap = False
            for r in best_list:
                r_bit = 1 << r
                mask_without_r = best_mask & ~r_bit
                # Find best candidate to add
                for a in range(m):
                    if a in best_list:
                        continue
                    if can_add(a, mask_without_r):
                        # Perform the swap
                        best_mask = mask_without_r | (1 << a)
                        best_list.remove(r)
                        best_list.append(a)
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                best_size = len(best_list)
                continue
            
            # Try 2-swap moves: remove r1, r2, add a1, a2
            found_2swap = False
            if len(best_list) >= 2:
                for i in range(len(best_list)):
                    if time.time() > deadline:
                        break
                    r1 = best_list[i]
                    r1_bit = 1 << r1
                    mask_without_r1 = best_mask & ~r1_bit
                    for j in range(i + 1, len(best_list)):
                        r2 = best_list[j]
                        r2_bit = 1 << r2
                        mask_without_r1_r2 = mask_without_r1 & ~r2_bit
                        # Try to find two elements to add
                        candidates_to_add = []
                        for a in range(m):
                            if a in best_list:
                                continue
                            if can_add(a, mask_without_r1_r2):
                                candidates_to_add.append(a)
                                if len(candidates_to_add) >= 2:
                                    break
                        if len(candidates_to_add) >= 2:
                            a1, a2 = candidates_to_add[0], candidates_to_add[1]
                            new_mask = mask_without_r1_r2 | (1 << a1) | (1 << a2)
                            # Verify no conflict between a1 and a2
                            if (adj_mask[a1] & (1 << a2)) == 0 and (adj_mask[a2] & (1 << a1)) == 0:
                                best_mask = new_mask
                                best_list.remove(r1)
                                best_list.remove(r2)
                                best_list.append(a1)
                                best_list.append(a2)
                                found_2swap = True
                                improved = True
                                break
                    if found_2swap:
                        break
            
            if found_2swap:
                best_size = len(best_list)
                continue
            
            # No improvement found; try removing an element and re-adding (kick move)
            if len(best_list) > 1:
                # Remove a random element and try to rebuild
                kick_idx = random.randint(0, len(best_list) - 1)
                kicked = best_list[kick_idx]
                kick_bit = 1 << kicked
                best_mask = best_mask & ~kick_bit
                best_list.remove(kicked)
                
                # Try to add elements back greedily
                for a in range(m):
                    if can_add(a, best_mask):
                        best_mask |= (1 << a)
                        best_list.append(a)
                        break
                continue
            
            # No moves available; break
            if not improved:
                break
        
        return best_list, best_mask

    # Multi-start local search
    best_solution_list = None
    best_solution_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # First, try deterministic orderings
    orders = [
        list(range(m)),
        list(range(m, 0, -1)),
        list(range(0, m, 2)) + list(range(1, m, 2)),
        list(range(1, m, 2)) + list(range(0, m, 2)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S_list, S_mask = greedy_init(order)
        S_list, S_mask = local_search(S_list, S_mask, max_iters=15000)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_solution_list = S_list
            best_solution_mask = S_mask
    
    # Now randomized restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Random permutation
        order = list(range(m))
        rng.shuffle(order)
        S_list, S_mask = greedy_init(order)
        S_list, S_mask = local_search(S_list, S_mask, max_iters=10000)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_solution_list = S_list
            best_solution_mask = S_mask
        
        # Also try: start from best, kick a random element, re-search
        if best_solution_list is not None and len(best_solution_list) > 2:
            if time.time() > deadline - 5:
                break
            S2_list = list(best_solution_list)
            S2_mask = best_solution_mask
            
            # Remove a random element
            idx = rng.randint(0, len(S2_list) - 1)
            removed = S2_list.pop(idx)
            S2_mask &= ~(1 << removed)
            
            # Try to add back different elements
            for a in range(m):
                if can_add(a, S2_mask):
                    S2_mask |= (1 << a)
                    S2_list.append(a)
                    break
            
            S2_list, S2_mask = local_search(S2_list, S2_mask, max_iters=8000)
            if len(S2_list) > best_size:
                best_size = len(S2_list)
                best_solution_list = S2_list
                best_solution_mask = S2_mask
    
    if best_solution_list is None:
        best_solution_list = [0]
    
    return np.array(sorted(best_solution_list), dtype=np.int64)


# EVOLVE-BLOCK-END
