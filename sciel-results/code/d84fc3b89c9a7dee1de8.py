import time
import sys

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # 1. Filter valid nodes (2*v not square)
    valid_nodes = []
    for v in range(1, 257):
        s = 2 * v
        r = int(s**0.5)
        if r*r != s and (r+1)*(r+1) != s: # Check square
            valid_nodes.append(v)
            
    n = len(valid_nodes)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict graph
    # conflict[i] is a bitmask of neighbors of i
    conflict = [0] * n
    
    # Squares up to 512
    squares = set()
    k = 2
    while k*k <= 512:
        squares.add(k*k)
        k += 1
        
    for i in range(n):
        vi = valid_nodes[i]
        for j in range(i+1, n):
            vj = valid_nodes[j]
            if vi + vj in squares:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    # 3. Branch and Bound for Maximum Independent Set
    best_size = 0
    best_mask = 0
    # Initial lower bound: Greedy
    # Simple greedy: pick node with min degree, add to set, remove neighbors
    def greedy_lower_bound(nodes_mask):
        size = 0
        mask = nodes_mask
        while mask:
            # Pick node with min degree in mask
            min_deg = n + 1
            min_node = -1
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = (conflict[idx] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_node = idx
                m ^= lsb
            size += 1
            mask &= ~(conflict[min_node] | (1 << min_node))
        return size

    initial_cands = (1 << n) - 1
    best_size = greedy_lower_bound(initial_cands)
    best_mask = 0 # We don't really need the mask for the score, just size? 
    # Wait, the task asks for the SET. "Return nonempty subset".
    # So we need to reconstruct the set.
    
    # To reconstruct, we need to store the choices.
    # Storing the path in B&B is memory intensive if deep.
    # Alternative: Store best_size, and at the end, run a greedy or another search to find a set of that size?
    # Or store the mask. n=245, mask is an int. Storing one int is fine.
    
    def solve(candidates, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > deadline:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Upper bound: Greedy Coloring
        # Returns number of colors
        def get_bound(cands):
            # Greedy coloring of the subgraph induced by cands
            # We want to color the graph such that adjacent nodes have different colors.
            # The number of colors is an upper bound on the MIS size.
            colors = 0
            remaining = cands
            while remaining:
                # Pick a vertex, e.g., first one
                lsb = remaining & -remaining
                v = lsb.bit_length() - 1
                # Color class: v and all non-neighbors of v in remaining
                # Actually, for the bound, we just need a valid coloring.
                # Greedy: assign color 'colors' to v and all vertices in remaining that are NOT adjacent to any vertex already in this color class?
                # No, standard greedy coloring:
                # Color class C is an independent set.
                # We form C by picking v, then picking u in remaining \ N[v] that is not adjacent to any in C?
                # Wait, if we pick u not adjacent to v, u might be adjacent to another w in C?
                # If we pick sequentially:
                # C = {v}
                # available = remaining \ (N[v] | C)
                # Pick u from available. Add to C.
                # available = available \ (N[u] | C)
                # ...
                # This forms an independent set C.
                # The number of such sets needed to cover the graph is the chromatic number.
                # So len(colors) is an upper bound on MIS.
                
                color_class = 0
                available = remaining & ~(conflict[v] | (1 << v))
                # Add v
                color_class |= (1 << v)
                # Add non-neighbors
                while available:
                    lsb_u = available & -available
                    u = lsb_u.bit_length() - 1
                    color_class |= (1 << u)
                    # Remove u and its neighbors from available
                    available &= ~(conflict[u] | (1 << u))
                
                colors += 1
                remaining &= ~color_class
                
            return colors

        bound = get_bound(candidates)
        if current_size + bound <= best_size:
            return
            
        # Pick a vertex to branch on
        # Heuristic: Max degree in candidates
        max_deg = -1
        v = -1
        m = candidates
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (conflict[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            m ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates minus v and its neighbors
        new_cands_include = candidates & ~(conflict[v] | (1 << v))
        solve(new_cands_include, current_size + 1, current_mask | (1 << v))
        
        # Branch 2: Exclude v
        new_cands_exclude = candidates & ~(1 << v)
        solve(new_cands_exclude, current_size, current_mask)
        
    solve(initial_cands, 0, 0)
    
    # Reconstruct set from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])
            
    if not result:
        # Fallback to seed if something went wrong
        return list(range(1, 257, 3))
        
    return result
