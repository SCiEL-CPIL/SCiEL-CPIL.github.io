import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

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
    
    # Precompute degrees for greedy
    degrees = [adj[u].bit_count() for u in range(m)]

    # Helper to check if a vertex can be added to a set represented by a bitmask
    def can_add(v, set_mask):
        if (set_mask >> v) & 1:
            return False
        return (adj[v] & set_mask) == 0

    def greedy_construct_min_degree(forbidden_mask=0):
        """
        Construct a maximal independent set using minimum degree heuristic.
        """
        current_mask = 0
        available = ((1 << m) - 1) & ~forbidden_mask
        # We want to pick vertices with low degree in the remaining graph
        # This is a heuristic approximation
        
        while available:
            # Find vertex in available with minimum degree to other available vertices
            best_v = -1
            best_deg = m + 1
            # Iterate through available vertices
            # Optimization: just iterate 0..m-1 and check bit
            for v in range(m):
                if (available >> v) & 1:
                    # Count neighbors in available
                    deg = (adj[v] & available).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            
            if best_v == -1:
                break
                
            current_mask |= (1 << best_v)
            # Remove best_v and its neighbors from available
            available &= ~(1 << best_v)
            available &= ~adj[best_v]
            
        return current_mask

    def greedy_construct_random(forbidden_mask=0):
        """
        Construct a maximal independent set using random selection.
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

    def local_search(initial_mask, max_iters=200):
        """
        Perform local search to improve the independent set.
        Uses 1-swap, 2-swap (remove 2 add 2), and 1-remove-2-add moves.
        """
        current_mask = initial_mask
        current_size = current_mask.bit_count()
        no_improvement_count = 0
        iterations = 0
        
        # Precompute list of vertices in set and complement for faster access?
        # Since m is small, we can just iterate.
        
        while iterations < max_iters and time.time() < deadline:
            iterations += 1
            improved = False
            
            set_verts = [i for i in range(m) if (current_mask >> i) & 1]
            comp_verts = [i for i in range(m) if not (current_mask >> i) & 1]
            
            if not set_verts or not comp_verts:
                break

            # 1. Try to add a valid vertex (net +1)
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

            # 2. Try 1-swap (net 0): Remove v, add u
            # This helps rearrange the set to allow future adds
            random.shuffle(set_verts)
            swapped = False
            for v in set_verts:
                # Candidates u must conflict with v and no one else in set
                # i.e., adj[u] & current_mask == (1 << v)
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
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                no_improvement_count = 0
                continue

            # 3. Try 1-remove-2-add (net +1)
            # Remove v, add u1, u2
            # u1, u2 must be valid in (current_mask \ {v})
            # And u1, u2 must not conflict with each other
            
            random.shuffle(set_verts)
            added2 = False
            for v in set_verts:
                new_mask = current_mask & ~(1 << v)
                # Find valid candidates in comp for new_mask
                valid_u = []
                for u in comp_verts:
                    if (adj[u] & new_mask) == 0:
                        valid_u.append(u)
                
                if len(valid_u) < 2:
                    continue
                
                # Try to find a pair u1, u2 in valid_u that don't conflict
                # To speed up, just check a few random pairs
                random.shuffle(valid_u)
                limit = min(len(valid_u), 20)
                found_pair = False
                for i in range(limit):
                    for j in range(i+1, limit):
                        u1 = valid_u[i]
                        u2 = valid_u[j]
                        if (adj[u1] >> u2) & 1:
                            continue
                        # Check if they conflict with each other? 
                        # adj[u1] & (1<<u2) == 0 is the check above.
                        # Also need to ensure they don't conflict with v? 
                        # No, v is removed.
                        # Also need to ensure u1, u2 are not in new_mask? 
                        # They are in comp, so they are not in current_mask, 
                        # and v is removed, so they are not in new_mask.
                        
                        # Perform move
                        current_mask = new_mask | (1 << u1) | (1 << u2)
                        current_size += 1
                        improved = True
                        added2 = True
                        found_pair = True
                        break
                    if found_pair:
                        break
                if added2:
                    break
            
            if added2:
                no_improvement_count = 0
                continue

            # 4. Try 2-swap (net 0): Remove v1, v2, add u1, u2
            # This is more complex and expensive. 
            # We might skip this in the main loop and only use it in a final polish 
            # or if we have plenty of time. Given m=205, O(m^4) is too slow.
            # We'll stick to 1-remove-2-add which is O(m^3) roughly if optimized, 
            # but our implementation is O(m^2 * K) which is okay for m=205.
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > 100:
                    break
                
        return current_mask, current_size

    best_mask = 0
    best_size = 0

    # Run multiple rounds
    # Start with min-degree greedy as it's usually stronger
    init_mask = greedy_construct_min_degree()
    new_mask, new_size = local_search(init_mask)
    if new_size > best_size:
        best_size = new_size
        best_mask = new_mask

    while time.time() < deadline - 10:
        # Alternate between random and min-degree? 
        # Or just random to explore different basins
        init_mask = greedy_construct_random()
        new_mask, new_size = local_search(init_mask)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask

    # Final polish: Run local search on best solution with more iterations
    if time.time() < deadline - 5:
        polished_mask, polished_size = local_search(best_mask, max_iters=500)
        if polished_size > best_size:
            best_mask = polished_mask
            best_size = polished_size

    # Extract result
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)

    if not result:
        result = [0]

    return np.array(result, dtype=np.int64)
