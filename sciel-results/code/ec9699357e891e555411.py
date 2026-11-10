import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Forbidden regular induced subgraphs on 6 vertices
    # Represented as frozensets of edges (u, v) with u < v
    forbidden_patterns = []
    
    # 0-regular: Empty
    forbidden_patterns.append(frozenset())
    
    # 1-regular: Perfect Matching
    forbidden_patterns.append(frozenset([(0,1), (2,3), (4,5)]))
    
    # 2-regular: C6
    forbidden_patterns.append(frozenset([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]))
    
    # 2-regular: C3 U C3
    forbidden_patterns.append(frozenset([(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]))
    
    # 3-regular: K3,3
    forbidden_patterns.append(frozenset([(i, j) for i in range(3) for j in range(3, 6)]))
    
    # 4-regular: Complement of Matching
    all_edges_6 = frozenset([(i, j) for i in range(6) for j in range(i+1, 6)])
    matching = frozenset([(0,1), (2,3), (4,5)])
    forbidden_patterns.append(all_edges_6 - matching)
    
    # 5-regular: K6
    forbidden_patterns.append(all_edges_6)

    # Precompute bitmasks for forbidden patterns on 6 vertices
    # We will map a 6-bit adjacency matrix to a canonical form or just compare sets
    # For speed, we can precompute the set of edge tuples for each pattern.
    
    def get_induced_edges(adj, vertices):
        """
        vertices: list of 6 vertex indices
        adj: list of int bitmasks
        Returns a frozenset of (u, v) local indices 0..5
        """
        edges = set()
        for i in range(6):
            for j in range(i+1, 6):
                if adj[vertices[i]] & (1 << vertices[j]):
                    edges.add((i, j))
        return frozenset(edges)

    def is_valid_graph(n, adj):
        """
        Check if graph with n vertices and adj list is valid.
        Valid means no induced subgraph of size >= 6 is regular.
        """
        # Iterate all subsets of size >= 6
        # Optimization: We can just iterate all masks.
        # For n=21, 2^21 is ~2M.
        # For each mask, if bit_count >= 6, check regularity.
        
        # To speed up, we can precompute bit counts? No, int.bit_count is fast.
        
        # We need to check if the induced subgraph is regular.
        # A subgraph is regular if all degrees are equal.
        
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Check if induced subgraph is regular
            # Get the vertices in the mask
            # Instead of extracting vertices list every time, we can compute degrees directly?
            # No, we need to compare degrees of all vertices in the subset.
            
            # Extract vertices
            verts = []
            m = mask
            v = 0
            while m:
                if m & 1:
                    verts.append(v)
                m >>= 1
                v += 1
            
            # Calculate degrees
            # deg_i = popcount(adj[verts[i]] & mask)
            degs = []
            for v_idx in verts:
                degs.append((adj[v_idx] & mask).bit_count())
            
            # Check if all degrees are equal
            if len(set(degs)) == 1:
                # It is regular. Is it one of the forbidden types?
                # The task says "No set of 6 or more vertices may induce a regular subgraph".
                # This means ANY regular induced subgraph of size >= 6 is forbidden.
                # So we don't need to match against the 6-vertex patterns specifically.
                # If a 7-vertex subgraph is 2-regular, it's forbidden.
                # So the check `len(set(degs)) == 1` is sufficient and necessary.
                return False
                
        return True

    # Search Strategy
    # Try n=21 first
    for n in [21, 20, 19]:
        if time.time() > deadline - 10:
            break
            
        found = False
        max_attempts = 5000 if n == 21 else 10000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if is_valid_graph(n, adj):
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Repair phase
            # If invalid, find a violating subset and flip edges
            # We need to modify is_valid_graph to return the violating subset
            # For now, we just restart. Random restarts might be slow.
            # Let's implement a repair step.
            
            # Find one violating subset
            violating_mask = None
            for mask in range(1, 1 << n):
                k = mask.bit_count()
                if k < 6:
                    continue
                verts = []
                m = mask
                v = 0
                while m:
                    if m & 1:
                        verts.append(v)
                    m >>= 1
                    v += 1
                
                degs = []
                for v_idx in verts:
                    degs.append((adj[v_idx] & mask).bit_count())
                
                if len(set(degs)) == 1:
                    violating_mask = mask
                    break
            
            if violating_mask is None:
                # Should not happen if is_valid_graph returned False
                continue
            
            # Flip a random edge in the violating subset
            verts = []
            m = violating_mask
            v = 0
            while m:
                if m & 1:
                    verts.append(v)
                m >>= 1
                v += 1
            
            # Pick two random vertices from the subset
            u, v = random.sample(verts, 2)
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Check if valid now
            if is_valid_graph(n, adj):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # If not, continue repair loop or restart
            # We'll do a few repair steps then restart
            # For simplicity in this draft, we restart. 
            # But repair is better. Let's add a small inner loop.
            
            # Inner repair loop
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                # Find violating subset again
                violating_mask = None
                for mask in range(1, 1 << n):
                    k = mask.bit_count()
                    if k < 6:
                        continue
                    verts = []
                    m = mask
                    v = 0
                    while m:
                        if m & 1:
                            verts.append(v)
                        m >>= 1
                        v += 1
                    degs = []
                    for v_idx in verts:
                        degs.append((adj[v_idx] & mask).bit_count())
                    if len(set(degs)) == 1:
                        violating_mask = mask
                        break
                
                if violating_mask is None:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                verts = []
                m = violating_mask
                v = 0
                while m:
                    if m & 1:
                        verts.append(v)
                    m >>= 1
                    v += 1
                
                u, v = random.sample(verts, 2)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                if is_valid_graph(n, adj):
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}

    # Fallback: return a known valid small graph
    # n=5 is valid
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
