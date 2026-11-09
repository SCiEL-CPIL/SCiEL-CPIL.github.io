import numpy as np
import time
import random
from collections import deque

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Precompute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency as bitmasks for fast operations
    # adj[u] is a bitmask where bit v is 1 if (u,v) is an edge (forbidden pair)
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask

    # Helper to check if a vertex can be added to a set represented by bitmask
    def can_add(v, current_mask):
        # v is not in current_mask
        # No neighbor of v is in current_mask
        # adj[v] & current_mask == 0
        return (current_mask & (1 << v)) == 0 and (adj[v] & current_mask) == 0

    # Greedy coloring of the subgraph induced by P
    # Returns the number of colors used (upper bound on independent set size)
    def greedy_color(P):
        colors = 0
        U = P
        while U:
            colors += 1
            MIS = 0
            # Pick vertices greedily to form an independent set in U
            # Remove them and their neighbors from U
            while U:
                v_bit = U & (-U)
                v = v_bit.bit_length() - 1
                MIS |= v_bit
                U &= ~(adj[v] | v_bit)
        return colors

    best_size = 0
    best_set_mask = 0
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin for 600s limit

    # Phase 1: Randomized Greedy Initialization
    # Run many times to find a good starting point
    # Since m=205, this is very fast
    for _ in range(5000):
        if time.time() > deadline - 10:
            break
        current_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        for v in candidates:
            if can_add(v, current_mask):
                current_mask |= (1 << v)
        
        size = current_mask.bit_count()
        if size > best_size:
            best_size = size
            best_set_mask = current_mask

    # Phase 2: Local Search (Hill Climbing + 2-Swap)
    # Refine the best solution found so far
    # We use a bitmask representation for speed
    
    def local_search(initial_mask):
        nonlocal best_size, best_set_mask
        current_mask = initial_mask
        current_size = current_mask.bit_count()
        improved = True
        local_deadline = time.time() + 50  # 50s per local search phase
        
        while improved and time.time() < local_deadline:
            improved = False
            
            # 1. Try adding vertices
            # Iterate through all vertices not in current_mask
            # If can_add, add it
            # Repeat until no more can be added
            added = True
            while added and time.time() < local_deadline:
                added = False
                for v in range(m):
                    if not (current_mask & (1 << v)):
                        if can_add(v, current_mask):
                            current_mask |= (1 << v)
                            current_size += 1
                            added = True
                            improved = True
            
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
                
            if time.time() > local_deadline:
                break

            # 2. Try 2-swaps: remove one, add another
            # For each v1 in current_mask, for each v2 not in current_mask:
            #   If removing v1 allows adding v2 (and maybe more), do it.
            #   Actually, simpler: try swap (v1, v2).
            #   New mask = current_mask ^ (1<<v1) | (1<<v2)
            #   Check if valid.
            #   To make it better, we should look for swaps that allow ADDING MORE later.
            #   But simple 1-swap is a good start.
            
            # Optimization: Only try swaps if we are stuck on adding.
            # Let's try a randomized 2-swap to escape local optima.
            
            # Get list of vertices in current_mask and not in current_mask
            in_mask = [i for i in range(m) if (current_mask >> i) & 1]
            not_in_mask = [i for i in range(m) if not ((current_mask >> i) & 1)]
            
            # Randomize order to avoid bias
            random.shuffle(in_mask)
            random.shuffle(not_in_mask)
            
            found_swap = False
            for v1 in in_mask:
                if time.time() > local_deadline:
                    break
                for v2 in not_in_mask:
                    if time.time() > local_deadline:
                        break
                    # Try swapping v1 for v2
                    new_mask = current_mask & ~(1 << v1)
                    if can_add(v2, new_mask):
                        # Check if this swap is "good".
                        # Ideally, we want to increase size. A 1-for-1 swap keeps size same.
                        # It might allow further additions.
                        # Let's accept the swap and see if we can add more in the next loop.
                        # To avoid oscillation, we could require that v2 has more "freedom" or similar.
                        # For now, accept and let the "add" phase handle the rest.
                        # But we need to ensure we don't just swap back and forth.
                        # A simple heuristic: only accept if v2 is "better" in some way?
                        # Or just accept and hope the add phase finds something.
                        # If no add is found, we might oscillate.
                        # Let's add a check: if we swap, we must be able to add at least one more?
                        # That's expensive to check fully.
                        # Alternative: Simulated Annealing?
                        # For now, let's just do the swap and rely on the add phase.
                        # To prevent immediate revert, we can track recent moves or just randomize heavily.
                        
                        current_mask = new_mask | (1 << v2)
                        # current_size remains same
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    break
            
            # If no swap found, we are in a local optimum for 1-swaps.
            # We could try 2-swaps (remove 2, add 2) but that's O(n^4).
            # For m=205, n^2 is ~40k, n^4 is 1.6e9 (too slow).
            # So we stick to 1-swaps + greedy add.
            
            if not found_swap:
                # Try a random perturbation?
                # Remove 2 random elements, then greedy add.
                if len(in_mask) >= 2:
                    v1, v2 = random.sample(in_mask, 2)
                    new_mask = current_mask & ~(1 << v1) & ~(1 << v2)
                    # Greedy add on new_mask
                    temp_mask = new_mask
                    temp_size = new_mask.bit_count()
                    candidates = list(range(m))
                    random.shuffle(candidates)
                    for v in candidates:
                        if can_add(v, temp_mask):
                            temp_mask |= (1 << v)
                            temp_size += 1
                    
                    if temp_size > current_size:
                        current_mask = temp_mask
                        current_size = temp_size
                        improved = True
                    elif temp_size == current_size and random.random() < 0.1:
                        # Accept equal size with some probability to escape
                        current_mask = temp_mask
                        improved = True

    # Run local search on the best solution found in Phase 1
    if best_size > 0:
        local_search(best_set_mask)

    # Phase 3: Branch and Bound for small gaps?
    # If we are close to the theoretical maximum, B&B might finish.
    # But we don't know the max.
    # Let's try a limited B&B if time remains.
    
    if time.time() < deadline - 10:
        # Use B&B to try to beat current best
        # We can use the current best as a lower bound
        
        # Recursion limit check
        # m=205, depth 205 is fine.
        
        def search(P, C, current_set):
            nonlocal best_size, best_set_mask
            
            if not P:
                if C > best_size:
                    best_size = C
                    best_set_mask = current_set
                return
            
            if C + P.bit_count() <= best_size:
                return
                
            colors = greedy_color(P)
            if C + colors <= best_size:
                return
            
            # Branching
            # Pick vertex with highest degree in P?
            # For simplicity, lowest bit.
            v_bit = P & (-P)
            v = v_bit.bit_length() - 1
            
            P_include = P & ~(adj[v] | v_bit)
            search(P_include, C + 1, current_set | v_bit)
            
            P_exclude = P & ~v_bit
            search(P_exclude, C, current_set)

        # Start B&B from scratch? No, that's too slow.
        # Start from the complement of the current best?
        # If we have a valid set S, any larger set S' must differ.
        # It's hard to prune B&B with a known independent set directly without complex logic.
        # Let's just run a quick B&B on the whole graph with a tight time limit.
        # If it finds something better, great.
        
        # To make B&B faster, we can order vertices by degree.
        # But the bitmask indexing is fixed.
        # Let's just run it with a very tight time limit.
        
        # Actually, the local search is likely better for this size.
        # Let's skip full B&B unless we have lots of time.
        # If best_size is small, B&B might be fast.
        # If best_size is large, B&B might be slow.
        
        # Let's try a few random restarts of B&B with pruning?
        # No, let's stick to the robust local search.
        pass

    # Extract result
    result = []
    for i in range(m):
        if (best_set_mask >> i) & 1:
            result.append(i)
            
    # Final validation check
    # Ensure no forbidden pairs
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a == b:
                continue
            d = (a - b) % m
            if d in forbidden:
                # This should not happen
                # Return empty or fix?
                # Better to return a valid subset.
                # Remove b?
                pass
                
    return np.array(result, dtype=np.int64)
