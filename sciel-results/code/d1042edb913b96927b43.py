import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Step 1: Generate all weight-4 supports (4-subsets of range(11))
    supports = list(itertools.combinations(range(D), 4))
    N = len(supports)  # 330
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Step 2: Build adjacency for the conflict graph J(11,4)
    # Two supports conflict if they intersect in exactly 3 elements.
    # We build a bitmask adjacency list.
    adj = [0] * N
    
    for i in range(N):
        s_i = set(supports[i])
        for j in range(i + 1, N):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Step 3: Find Maximum Independent Set using Branch and Bound
    # Greedy coloring bound for B&B
    
    def greedy_color(candidates_mask):
        # Returns number of colors needed for the subgraph induced by candidates_mask
        # This is an upper bound on the MIS size of that subgraph.
        # Simple greedy: pick node, remove neighbors, count 1, repeat.
        # Better: Sort by degree, then color.
        # For B&B bound, a fast greedy is needed.
        
        colors = 0
        remaining = candidates_mask
        while remaining:
            colors += 1
            # Pick a node with high degree in remaining to remove more?
            # Or just pick the first one for speed?
            # Let's pick the one with the most neighbors in 'remaining' to reduce search space?
            # Actually, for coloring bound, we want to pack as many as possible into one color class.
            # A color class is an independent set.
            # So we want a maximal independent set in 'remaining'.
            # Greedy: pick node, remove it and its neighbors from 'remaining'.
            
            # Find a node in remaining
            # To make the bound tighter, we should pick a node that "covers" many others?
            # No, for coloring, we want the color class to be large.
            # So we want to pick a node that has FEW neighbors in the current color class?
            # Standard greedy coloring:
            # 1. Order nodes.
            # 2. Assign smallest possible color.
            # This is slow.
            
            # Alternative bound: Use a precomputed greedy independent set size?
            # Let's use a simple heuristic: 
            # Pick node v with max degree in remaining.
            # Add v to color class.
            # Remove v and N(v) from remaining.
            
            # Find node with max degree in remaining
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                # Count neighbors in remaining
                deg = (adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp &= temp - 1
                
            if best_v == -1:
                break
                
            # Remove best_v and its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
            
        return colors

    best_size = 0
    best_indices = []
    
    def branch_and_bound(candidates_mask, current_size, current_set):
        nonlocal best_size, best_indices
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_indices = current_set[:]
            return
            
        # Bound
        bound = greedy_color(candidates_mask)
        if current_size + bound <= best_size:
            return
            
        # Select a node to branch on
        # Heuristic: Pick node with max degree in candidates
        v = -1
        max_deg = -1
        temp = candidates_mask
        while temp:
            node = (temp & -temp).bit_length() - 1
            deg = (adj[node] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = node
            temp &= temp - 1
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_mask_include = candidates_mask & ~(1 << v) & ~adj[v]
        current_set.append(v)
        branch_and_bound(new_mask_include, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_mask_exclude = candidates_mask & ~(1 << v)
        branch_and_bound(new_mask_exclude, current_size, current_set)
        
    # Initial candidates: all nodes
    initial_mask = (1 << N) - 1
    branch_and_bound(initial_mask, 0, [])
    
    # Step 4: Construct vectors
    if best_indices:
        vectors = []
        for idx in best_indices:
            v = np.zeros(D, dtype=np.int64)
            support = supports[idx]
            v[support[0]] = 1
            v[support[1]] = 1
            v[support[2]] = 1
            v[support[3]] = 1
            vectors.append(v)
        return np.array(vectors)
    else:
        # Fallback
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
