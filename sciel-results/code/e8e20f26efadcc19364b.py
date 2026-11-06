import time
import random
import math

def construct():
    t0 = time.time()
    # Safety margin: 90s limit, aim for 80s
    DEADLINE = t0 + 80.0
    
    N = 256
    
    # Precompute conflict masks using Python integers as bitsets
    # conflict_mask[i] is a bitmask of j such that (i+1, j+1) is a conflicting pair
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Precompute neighbor lists for faster operations
    neighbors = []
    for i in range(N):
        nbrs = set()
        m = conflict_mask[i]
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            nbrs.add(j)
            m ^= lsb
        neighbors.append(nbrs)
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_size(mask):
        return popcount(mask)
    
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0

    def valid_set(mask):
        indices = [i for i in range(N) if (mask >> i) & 1]
        for ii in range(len(indices)):
            for jj in range(ii+1, len(indices)):
                a = indices[ii]
                b = indices[jj]
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True

    # Tomita-style Maximum Independent Set via Branch and Bound
    # Using greedy coloring for upper bound
    
    best_mask = 0
    best_size = 0
    
    # First, get a good initial solution via greedy
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Greedy: start with all odds, then add compatible evens
    mask_odds = 0
    for i in odds_indices:
        mask_odds |= (1 << i)
    
    current_mask = mask_odds
    for i in evens_indices:
        if can_add(i, current_mask):
            current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = get_size(current_mask)
    
    # Tomita MIS algorithm
    # We work with a candidate set (bitset) and try to find an independent set
    # within it, combined with the current solution
    
    def greedy_color(candidates, conflict_masks):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns a list of colors (one per vertex in candidates, in order).
        The number of distinct colors used is an upper bound on the
        maximum independent set size within candidates.
        """
        # Order vertices by degree in the induced subgraph (descending)
        # For efficiency, we'll use a simple ordering
        
        vertices = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if not vertices:
            return []
        
        # Compute degrees within the candidate set
        deg = {}
        for v in vertices:
            d = popcount(conflict_masks[v] & candidates)
            deg[v] = d
        
        # Sort by degree descending
        vertices.sort(key=lambda v: -deg[v])
        
        # Greedy coloring: assign smallest possible color
        color_of = {}
        color_class = {}  # color -> set of vertices
        next_color = 0
        
        for v in vertices:
            # Find colors used by neighbors of v in candidates
            used_colors = set()
            nbr_mask = conflict_masks[v] & candidates
            while nbr_mask:
                lsb = nbr_mask & (-nbr_mask)
                u = lsb.bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                nbr_mask ^= lsb
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c not in color_class:
                color_class[c] = set()
            color_class[c].add(v)
            next_color = max(next_color, c + 1)
        
        return color_of, next_color
    
    def tomita_mis(candidates, conflict_masks, deadline):
        """
        Tomita-style branch and bound for Maximum Independent Set.
        candidates: bitset of candidate vertices
        Returns the best independent set found within candidates (as bitset).
        """
        global best_mask, best_size
        
        if time.time() > deadline:
            return 0
        
        if not candidates:
            return 0
        
        # Get upper bound via greedy coloring
        color_of, num_colors = greedy_color(candidates, conflict_masks)
        
        # Upper bound on additional vertices we can add
        ub = num_colors
        
        # If upper bound doesn't improve on current best, prune
        # (This is a simplified check; the full Tomita uses color classes)
        
        # Select a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidate subgraph
        vertices = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if not vertices:
            return 0
        
        # Pick vertex with max degree
        deg = {}
        for v in vertices:
            d = popcount(conflict_masks[v] & candidates)
            deg[v] = d
        
        v = max(vertices, key=lambda x: deg[x])
        
        # Branch 1: Include v
        # Remove v and its neighbors from candidates
        include_mask = 1 << v
        remaining = candidates & ~(conflict_masks[v] | (1 << v))
        
        # Recursively find MIS in remaining
        sub_best = tomita_mis(remaining, conflict_masks, deadline)
        candidate1 = include_mask | sub_best
        size1 = get_size(candidate1)
        
        # Branch 2: Exclude v
        remaining2 = candidates & ~(1 << v)
        sub_best2 = tomita_mis(remaining2, conflict_masks, deadline)
        candidate2 = sub_best2
        size2 = get_size(candidate2)
        
        if size1 >= size2:
            return candidate1
        else:
            return candidate2
    
    # Run Tomita MIS on the full graph
    full_candidates = (1 << N) - 1
    deadline = DEADLINE - 5.0
    
    # To make Tomita more effective, we use it to improve on our initial solution
    # We'll try to find a larger independent set
    
    # First, let's try running Tomita on the full candidate set
    # But this might be too slow for N=256, so we'll use it strategically
    
    # Alternative: Use Tomita on subsets to improve local search
    # For now, let's do a simple branch and bound with the initial solution as lower bound
    
    def bb_mis(candidates, current_size, deadline):
        """
        Branch and bound to find MIS in candidates, 
        trying to beat current_size.
        Returns best independent set found.
        """
        global best_mask, best_size
        
        if time.time() > deadline:
            return 0
        
        if not candidates:
            return 0
        
        # Upper bound via greedy coloring
        color_of, num_colors = greedy_color(candidates, conflict_mask)
        
        if num_colors <= current_size:
            return 0  # Can't beat current
        
        # Select vertex to branch on
        vertices = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if not vertices:
            return 0
        
        # Pick vertex with max degree
        deg = {}
        for v in vertices:
            d = popcount(conflict_mask[v] & candidates)
            deg[v] = d
        
        v = max(vertices, key=lambda x: deg[x])
        
        # Branch 1: Include v
        include_mask = 1 << v
        remaining = candidates & ~(conflict_mask[v] | (1 << v))
        
        # The remaining can contribute at most num_colors - 1 more (rough bound)
        sub_best = bb_mis(remaining, current_size - 1, deadline)
        candidate1 = include_mask | sub_best
        size1 = get_size(candidate1)
        
        if size1 > current_size:
            current_size = size1
        
        # Branch 2: Exclude v
        remaining2 = candidates & ~(1 << v)
        sub_best2 = bb_mis(remaining2, current_size, deadline)
        candidate2 = sub_best2
        size2 = get_size(candidate2)
        
        if size1 >= size2:
            return candidate1
        else:
            return candidate2
    
    # Run B&B to try to improve on our initial solution
    # We'll run it on the full set with our current best as the lower bound
    deadline_bb = DEADLINE - 10.0
    
    # To avoid infinite recursion issues, let's use an iterative approach
    # or limit the depth
    
    # Actually, for N=256, pure recursive B&B may be too slow.
    # Let's use a hybrid: run B&B on smaller subsets or use it to guide local search.
    
    # Strategy: Use the coloring-based upper bound to guide a better local search
    # Or: Run B&B with a time limit and return whatever it finds
    
    # Let's try a modified approach: use B&B but with aggressive pruning
    # and a time check at each level
    
    def bb_mis_limited(candidates, current_best_size, deadline, depth=0):
        """
        Limited depth B&B with time checking.
        """
        global best_mask, best_size
        
        if time.time() > deadline:
            return 0
        
        if not candidates:
            return 0
        
        # Upper bound
        color_of, num_colors = greedy_color(candidates, conflict_mask)
        
        if num_colors <= current_best_size:
            return 0
        
        # If small enough, solve exactly by brute force
        cand_size = popcount(candidates)
        if cand_size <= 20:
            # Brute force
            vertices = []
            m = candidates
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                vertices.append(v)
                m ^= lsb
            
            best_local = 0
            best_local_size = 0
            
            # Try all subsets? No, that's 2^20 which is fine
            # Actually, let's just do greedy + local improvement
            # Or use a simple recursive MIS for small sets
            
            def small_mis(cands):
                if not cands:
                    return 0
                v = (cands & (-cands)).bit_length() - 1
                # Include v
                rem1 = cands & ~(conflict_mask[v] | (1 << v))
                sol1 = (1 << v) | small_mis(rem1)
                # Exclude v
                rem2 = cands & ~(1 << v)
                sol2 = small_mis(rem2)
                if popcount(sol1) >= popcount(sol2):
                    return sol1
                else:
                    return sol2
            
            result = small_mis(candidates)
            return result
        
        # Select vertex to branch on
        vertices = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        if not vertices:
            return 0
        
        # Pick vertex with max degree
        deg = {}
        for v in vertices:
            d = popcount(conflict_mask[v] & candidates)
            deg[v] = d
        
        v = max(vertices, key=lambda x: deg[x])
        
        # Branch 1: Include v
        include_mask = 1 << v
        remaining = candidates & ~(conflict_mask[v] | (1 << v))
        
        sub_best = bb_mis_limited(remaining, current_best_size - 1, deadline, depth + 1)
        candidate1 = include_mask | sub_best
        size1 = get_size(candidate1)
        
        if size1 > current_best_size:
            current_best_size = size1
        
        # Branch 2: Exclude v
        remaining2 = candidates & ~(1 << v)
        sub_best2 = bb_mis_limited(remaining2, current_best_size, deadline, depth + 1)
        candidate2 = sub_best2
        size2 = get_size(candidate2)
        
        if size1 >= size2:
            return candidate1
        else:
            return candidate2
    
    # Run B&B
    deadline_bb = DEADLINE - 15.0
    bb_result = bb_mis_limited(full_candidates, best_size, deadline_bb)
    
    if get_size(bb_result) > best_size:
        best_mask = bb_result
        best_size = get_size(bb_result)
    
    # Final polish: try to add any node that fits
    for i in range(N):
        if not ((best_mask >> i) & 1) and can_add(i, best_mask):
            best_mask |= (1 << i)
    
    if not valid_set(best_mask):
        # Fallback to all odds
        best_mask = 0
        for i in odds_indices:
            best_mask |= (1 << i)
    
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
