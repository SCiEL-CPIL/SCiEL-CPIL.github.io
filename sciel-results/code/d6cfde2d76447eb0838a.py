import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill

    # 1. Precompute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)

    # 2. Build adjacency bitmasks: adj[u] has bit v set iff u and v conflict
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask

    # Helper to check if a vertex can be added to a set represented by a bitmask
    def can_add(v, set_mask):
        if (set_mask >> v) & 1:
            return False
        return (adj[v] & set_mask) == 0

    def greedy_construct(forbidden_mask=0):
        """
        Construct a maximal independent set.
        forbidden_mask: vertices that cannot be added.
        """
        current_mask = 0
        available = ((1 << m) - 1) & ~forbidden_mask
        while available:
            # Pick a random vertex from available
            found = False
            while not found:
                v = random.randint(0, m - 1)
                if (available >> v) & 1:
                    found = True
                    
            current_mask |= (1 << v)
            available &= ~(1 << v)
            available &= ~adj[v]
            
        return current_mask

    def local_search(initial_mask, max_iters=1000):
        """
        Perform local search to improve the independent set.
        Uses 1-add, 1-swap, and 2-swap moves.
        """
        current_mask = initial_mask
        current_size = current_mask.bit_count()
        no_improvement_count = 0
        iterations = 0
        
        while iterations < max_iters and time.time() < deadline - 5:
            iterations += 1
            improved = False
            
            # Get list of vertices in set and not in set
            set_verts = [i for i in range(m) if (current_mask >> i) & 1]
            comp_verts = [i for i in range(m) if not (current_mask >> i) & 1]
            
            if not set_verts or not comp_verts:
                break

            # 1. Try to add a valid vertex from complement
            random.shuffle(comp_verts)
            added = False
            for u in comp_verts:
                if can_add(u, current_mask):
                    current_mask |= (1 << u)
                    current_size += 1
                    improved = True
                    added = True
                    break
            
            if added:
                no_improvement_count = 0
                continue
            
            # 2. Try 1-swap: Remove v, add u
            random.shuffle(set_verts)
            swapped = False
            for v in set_verts:
                # Candidates for u must be neighbors of v in comp
                # AND v must be the only conflict for u in current_mask
                candidates = []
                for u in comp_verts:
                    if (adj[v] >> u) & 1:
                        conflicts = adj[u] & current_mask
                        if conflicts == (1 << v):
                            candidates.append(u)
                
                if candidates:
                    u = random.choice(candidates)
                    current_mask &= ~(1 << v)
                    current_mask |= (1 << u)
                    # Size remains same, but structure improved potentially
                    # We don't count this as "improvement" in size, but it allows future adds
                    # However, standard local search usually requires size increase for "improvement"
                    # Let's treat swap as a move that resets no_improvement only if it leads to add?
                    # No, let's just do the swap and continue. It breaks local optima of "no adds".
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                no_improvement_count = 0
                continue

            # 3. Try 2-swap: Remove v1, v2, add u1, u2 (net size +0, but might allow +1 later)
            # Or Remove 1, Add 2 (net size +1)
            # Let's try: Remove v, Add u1, u2
            # This is expensive. Let's try a simpler 2-opt:
            # Remove v1, v2. Add u1, u2.
            # Check if {u1, u2} is independent and independent from (current_mask \ {v1, v2})
            
            # Optimization: Only try 2-swaps if we've been stuck for a while
            if no_improvement_count > 20:
                # Pick random pair from set
                if len(set_verts) >= 2:
                    random.shuffle(set_verts)
                    v1, v2 = set_verts[0], set_verts[1]
                    # Try to find two vertices in comp that can replace them
                    # This is O(N^2) per pair.
                    # Let's just try a few random pairs from comp
                    random.shuffle(comp_verts)
                    found_2swap = False
                    for i in range(min(10, len(comp_verts))):
                        u1 = comp_verts[i]
                        for j in range(i+1, min(10, len(comp_verts))):
                            u2 = comp_verts[j]
                            # Check u1, u2 independent
                            if (adj[u1] >> u2) & 1:
                                continue
                            # Check u1, u2 against current_mask without v1, v2
                            new_mask_test = current_mask & ~(1 << v1) & ~(1 << v2)
                            if (adj[u1] & new_mask_test) == 0 and (adj[u2] & new_mask_test) == 0:
                                # Also need to ensure u1, u2 are not v1, v2 (they are in comp, so ok)
                                current_mask = new_mask_test
                                current_mask |= (1 << u1)
                                current_mask |= (1 << u2)
                                # Size is same, but we moved.
                                improved = True
                                found_2swap = True
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        no_improvement_count = 0
                        continue

            no_improvement_count += 1
            if no_improvement_count > 100:
                break
                
        return current_mask, current_size

    best_mask = 0
    best_size = 0

    # Run multiple rounds of Greedy + Local Search
    while time.time() < deadline - 20:
        # Randomized Greedy
        init_mask = greedy_construct()
        init_size = init_mask.bit_count()
        
        # Local Search
        new_mask, new_size = local_search(init_mask)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask

    # Final check: If we still have time, try to polish the best solution
    if time.time() < deadline - 10:
        # Run local search again on the best found
        polished_mask, polished_size = local_search(best_mask, max_iters=5000)
        if polished_size > best_size:
            best_mask = polished_mask
            best_size = polished_size
            
    # One more specific attempt: try to add to best_mask
    if time.time() < deadline - 5:
        final_mask, final_size = local_search(best_mask, max_iters=1000)
        if final_size > best_size:
            best_mask = final_mask
            best_size = final_size

    # Extract result
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)

    if not result:
        # Fallback to a single valid element
        result = [0]

    return np.array(result, dtype=np.int64)
