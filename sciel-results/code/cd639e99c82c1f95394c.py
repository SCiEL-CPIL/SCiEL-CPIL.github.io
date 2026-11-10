import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    p1, p2 = 5, 41
    
    # 1. Compute forbidden differences (non-zero quadratic residues mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitsets for G (edges where diff is in forbidden)
    # adj[i] has bit j set if (i-j) mod 205 is in forbidden
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        adj[i] = mask
    
    # 3. Precompute a static ordering by degree (highest degree first for B&B)
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    order = list(range(m))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # 4. CRT-based construction to get a good initial lower bound
    # Z_205 ≅ Z_5 × Z_41
    # d is a non-zero square mod 205 iff d mod 5 is a non-zero square mod 5 
    #   AND d mod 41 is a non-zero square mod 41? 
    # Actually: d is a square mod 205 iff d is a square mod 5 AND d is a square mod 41.
    # Non-zero squares mod 5: {1, 4}
    # Non-zero squares mod 41: {1, 4, 5, 9, 16, 25, 36}
    # Wait, let's just use the direct forbidden set for correctness.
    
    # Strategy for initial solution: Greedy with random restarts
    def greedy_random():
        # Random permutation
        perm = list(range(m))
        np.random.shuffle(perm)
        selected = []
        for v in perm:
            # Check if v is compatible with all selected
            ok = True
            for u in selected:
                if (v - u) % m in forbidden or (u - v) % m in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(v)
        return selected
    
    # Try a few random greedy to get a decent starting point
    best_size = 0
    best_set = 0
    best_list = []
    
    for _ in range(100):
        s = greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_list = s
            best_set = 0
            for v in s:
                best_set |= (1 << v)
    
    # Also try a deterministic greedy (by degree)
    s_det = []
    for v in order:
        ok = True
        for u in s_det:
            if (v - u) % m in forbidden or (u - v) % m in forbidden:
                ok = False
                break
        if ok:
            s_det.append(v)
    if len(s_det) > best_size:
        best_size = len(s_det)
        best_list = s_det
        best_set = 0
        for v in s_det:
            best_set |= (1 << v)
    
    # 5. Branch and Bound with better upper bound
    # Upper bound: Greedy coloring of the induced subgraph on candidates
    # We want the size of the maximum independent set.
    # An upper bound is the chromatic number of the complement graph (which is the clique number of the original graph? No).
    # The complement graph G' has edges where difference is NOT forbidden.
    # We are looking for a clique in G'.
    # Upper bound on clique size in G' is the chromatic number of G (the original graph)? No.
    # Upper bound on independent set in G is the chromatic number of G.
    # So we need to compute a lower bound on the chromatic number of G[candidates].
    # A simple lower bound on chromatic number is the size of the largest clique in G[candidates].
    # Or we can use a greedy coloring of G[candidates] to get an upper bound on the chromatic number,
    # which is an upper bound on the independent set size? No, chromatic number is an upper bound on independent set size?
    # No. The chromatic number χ(G) is at least the size of the maximum independent set? No.
    # α(G) ≤ n / ω(G)? No.
    # α(G) ≤ χ(G). So if we find a coloring with k colors, then α(G) ≤ k.
    # So we need to color the graph G[candidates].
    
    def greedy_color(candidates):
        if candidates == 0:
            return 0
        colors = 0
        remaining = candidates
        while remaining:
            colors += 1
            # Greedily pick a vertex, add it to current color class, remove its neighbors from remaining
            # We want to color G, so adjacent vertices cannot share a color.
            # So in one color class, we pick a set of vertices with no edges between them in G.
            # This is a maximal independent set in G.
            color_class = 0
            compatible = remaining
            # Pick vertices that are not adjacent to each other
            # To maximize the size of the color class (to minimize total colors), we should pick well-connected vertices?
            # No, we just need a valid coloring.
            # Standard greedy: pick a vertex, add to color class, remove it and its neighbors from 'available for this color'?
            # No, if I pick v for color c, I cannot pick any neighbor of v for color c.
            # So I remove v and all neighbors of v from the set of vertices available for color c.
            # But I only remove v from 'remaining'.
            
            # Let's re-phrase:
            # We are coloring G.
            # Step 1: Pick a vertex v from remaining.
            # Step 2: Assign color c to v.
            # Step 3: Remove v from remaining.
            # Step 4: For the rest of the vertices in this color class, they must not be adjacent to v.
            # So we restrict the candidates for this color class to remaining \ N(v).
            
            # Let's implement a simple greedy coloring:
            # 1. Find a maximal independent set in G[remaining].
            # 2. Assign current color to all vertices in this set.
            # 3. Remove these vertices from remaining.
            
            # To find a maximal independent set, we can greedily pick vertices.
            # To get a good upper bound (tight), we want the coloring to use as few colors as possible.
            # This is hard. A simple greedy coloring often works well enough.
            
            # Let's try: Pick a vertex v (e.g., highest degree in remaining).
            # Add v to color class.
            # Remove v and all neighbors of v from the set of vertices that can take this color.
            # But we only remove v from 'remaining'.
            
            # Actually, the standard DSATUR or simple greedy coloring is:
            # For each vertex in some order:
            #   Assign the smallest color not used by its neighbors.
            
            # This is easier to implement and gives a valid upper bound.
            # However, the order matters.
            
            # Let's stick to the "Maximal Independent Set" approach for the color class.
            # This is equivalent to finding a maximal independent set in G[remaining].
            # We can use the same greedy logic as before.
            
            v = -1
            # Pick vertex with max degree in remaining?
            max_deg = -1
            for u in order:
                if (remaining >> u) & 1:
                    deg = bin(adj[u] & remaining).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        v = u
            if v == -1:
                break
            
            # Start color class with v
            color_class |= (1 << v)
            # Vertices that can still be added to this color class:
            # They must be in remaining, and not adjacent to any vertex in color_class.
            available = remaining & ~(1 << v)
            for u in order:
                if (color_class >> u) & 1:
                    available &= ~adj[u]
            
            # Now greedily add vertices from available to color_class
            # To make the color class large, we should pick vertices that have many neighbors in available?
            # Or just pick the first one?
            # Let's pick the one with max degree in available.
            while available:
                u = -1
                max_deg = -1
                for w in order:
                    if (available >> w) & 1:
                        deg = bin(adj[w] & available).count('1')
                        if deg > max_deg:
                            max_deg = deg
                            u = w
                if u == -1:
                    break
                color_class |= (1 << u)
                available &= ~(adj[u] | (1 << u))
            
            remaining &= ~color_class
        return colors

    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        ub = greedy_color(candidates)
        if current_size + ub <= best_size:
            return
        
        v = -1
        for u in order:
            if (candidates >> u) & 1:
                v = u
                break
        
        if v == -1:
            return
        
        # Include v
        include_candidates = candidates & ~(adj[v] | (1 << v))
        branch_and_bound(include_candidates, current_set | (1 << v), current_size + 1)
        
        # Exclude v
        exclude_candidates = candidates & ~(1 << v)
        branch_and_bound(exclude_candidates, current_set, current_size)

    deadline = time.time() + 590
    all_mask = (1 << m) - 1
    
    # Run B&B
    branch_and_bound(all_mask, 0, 0)
    
    result = []
    for i in range(m):
        if (best_set >> i) & 1:
            result.append(i)
            
    if not result:
        result = [0]
        
    return np.array(result, dtype=np.int64)
