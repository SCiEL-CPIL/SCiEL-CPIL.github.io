import time
import itertools

def construct():
    start_time = time.time()
    
    # Candidate generators
    def make_paley(q):
        squares = {x * x % q for x in range(1, q)}
        n = q
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in squares:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    def is_valid(graph):
        n = graph['n']
        edges = graph['edges']
        
        # Build adjacency set for fast lookup
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check K4
        # For each edge (i,j), check if N(i) \cap N(j) contains an edge.
        has_k4 = False
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    # Find common neighbors
                    common = adj[u].intersection(adj[v])
                    # Check if common contains an edge
                    for x in common:
                        if any(y in adj[x] for y in common if y > x):
                            has_k4 = True
                            break
                if has_k4: break
            if has_k4: break
        
        if has_k4:
            return False

        # Check Independent Set of size 6
        # Use backtracking with bitsets for speed
        
        # Precompute adjacency as bitmasks for faster independent set check
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        def has_independent_set(k, start=0):
            # Backtracking to find independent set of size k
            # Using bitmasks for available vertices
            # We want to find k vertices such that no two are adjacent
            
            # available: bitmask of vertices that can still be added
            # We process vertices in order
            
            def rec(remaining_count, available_mask, last_vertex):
                if remaining_count == 0:
                    return True
                if available_mask == 0:
                    return False
                
                # Pruning: if number of available vertices < remaining_count
                # popcount check
                if bin(available_mask).count('1') < remaining_count:
                    return False
                
                # Iterate through available vertices
                # We can pick any vertex v from available_mask
                # But to avoid duplicates, we should enforce an ordering
                # Standard technique: pick the lowest set bit, or iterate bits
                
                # Let's iterate through bits in available_mask
                # For each vertex v, if we pick it, we remove it and all its neighbors from available
                # We also must ensure we don't pick vertices <= last_vertex to maintain order?
                # Actually, if we just remove neighbors and the vertex itself, and recurse,
                # we might pick {1, 2} and {2, 1}.
                # To enforce order, we can restrict the search space.
                # If we pick v, the next vertices must be > v.
                # So we can mask out all vertices <= v.
                
                # Let's try:
                # Pick v (lowest bit first? or any?)
                # If we pick v, new_available = available_mask & ~(1 << v) & ~adj_mask[v]
                # And we also need to ensure we only pick vertices > v?
                # If we do that, we cover all combinations exactly once.
                
                # Let's find the lowest set bit
                # Or just iterate through all bits?
                # For efficiency, picking the lowest bit first might help pruning?
                
                # Let's iterate through bits in available_mask
                mask = available_mask
                while mask:
                    # Get lowest set bit
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    mask ^= lsb
                    
                    # If we pick v
                    # New available: remove v, remove neighbors of v
                    # Also remove all vertices <= v to enforce strict increasing order
                    # Why? Because if we pick v, any future pick must be > v
                    # This ensures we generate each set exactly once.
                    
                    # Mask for vertices > v
                    # Vertices 0..v-1 are disallowed
                    # So we want to clear bits 0..v
                    # mask_gt_v = available_mask & ~((1 << (v+1)) - 1)
                    # But we also need to remove neighbors of v from mask_gt_v
                    
                    # Actually, simpler:
                    # new_available = (available_mask & ~(1 << v)) & ~adj_mask[v]
                    # Then, to enforce order, we only consider vertices > v in the next step.
                    # So new_available = new_available & ~((1 << (v+1)) - 1)
                    
                    # Let's combine:
                    # Candidates are in available_mask.
                    # If we pick v, next candidates are in available_mask, excluding v, excluding N(v), and excluding {0..v}.
                    
                    # mask_excluding_v_and_below = available_mask & ~((1 << (v+1)) - 1)
                    # This keeps only vertices > v.
                    # Then remove neighbors of v.
                    # new_avail = mask_excluding_v_and_below & ~adj_mask[v]
                    
                    # Wait, adj_mask[v] includes neighbors which might be < v or > v.
                    # We only care about neighbors > v because we are restricting to > v.
                    # So intersecting with ~adj_mask[v] is fine.
                    
                    new_avail = (available_mask & ~((1 << (v+1)) - 1)) & ~adj_mask[v]
                    
                    if rec(remaining_count - 1, new_avail, v):
                        return True
                
                return False
            
            # Initial available: all vertices
            # But we must enforce order from the start.
            # So we can just start with all vertices and let the recursion handle the ordering constraint.
            # The recursion above assumes that when we pick v, we restrict to > v.
            # So the initial call should pass all vertices.
            return rec(k, (1 << n) - 1, -1)
        
        if has_independent_set(6):
            return False
            
        return True

    # Try candidates
    valid_graphs = []
    
    # Test Paley 17
    g17 = make_paley(17)
    if is_valid(g17):
        valid_graphs.append(g17)
        
    # Test Paley 29
    g29 = make_paley(29)
    if is_valid(g29):
        valid_graphs.append(g29)
        
    # Test Paley 37
    g37 = make_paley(37)
    if is_valid(g37):
        valid_graphs.append(g37)
        
    # Test Paley 41 (n=41 > 40, so invalid per contract, but let's check validity first)
    # Contract says 1 <= n <= 40. So 41 is not allowed.
    # But wait, the task says "Maximize n. A valid graph on n vertices certifies R(4,6) > n."
    # And "1 <= n <= 40".
    # So we should only return graphs with n <= 40.
    
    if valid_graphs:
        # Filter by n <= 40
        valid_graphs = [g for g in valid_graphs if g['n'] <= 40]
        if valid_graphs:
            return max(valid_graphs, key=lambda x: x['n'])
    
    # Fallback: return the seed
    return g17
