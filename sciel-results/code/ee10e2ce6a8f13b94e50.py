import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def is_triangle_free(n, adj):
        """Check triangle-free using adjacency bitsets."""
        for u in range(n):
            neighbors = adj[u]
            # For each pair of neighbors, check if they're connected
            v_bit = neighbors
            while v_bit:
                v = (v_bit & -v_bit).bit_length() - 1
                v_bit &= v_bit - 1
                # Check if u and v share a common neighbor
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return False
        return True

    def has_independent_set_10(n, adj):
        """Branch and bound for independent set of size 10 using bitsets."""
        
        def backtrack(candidates, size):
            if size == 10:
                return True
            # Pruning: if size + popcount(candidates) < 10, return False
            if size + bin(candidates).count('1') < 10:
                return False
            
            if candidates == 0:
                return False
            
            # Heuristic: pick vertex with highest degree in candidate subgraph
            # This tends to prune more effectively
            best_v = -1
            best_deg = -1
            v_bit = candidates
            while v_bit:
                v = (v_bit & -v_bit).bit_length() - 1
                v_bit &= v_bit - 1
                deg = bin(adj[v] & candidates).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            candidates &= ~(1 << v)
            
            # Option 1: Include v
            new_candidates = candidates & ~adj[v]
            if backtrack(new_candidates, size + 1):
                return True
            
            # Option 2: Exclude v
            if backtrack(candidates, size):
                return True
                
            return False
        
        return backtrack((1 << n) - 1, 0)

    def validate(n, edges):
        """Validate a graph is triangle-free and has no independent set of 10."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if not is_triangle_free(n, adj):
            return False
        if has_independent_set_10(n, adj):
            return False
        return True

    # Known strong baseline: Andrássy graph on 28 vertices
    # This is a triangle-free graph with independence number 9
    best_n = 28
    # Construct Andrássy graph A_28
    # It's a blow-up of C_5 with specific structure
    # Simpler: use a known construction
    # For now, use a random seed to generate a valid 28-vertex graph
    # Actually, let's use a deterministic construction
    
    # Andrássy graph on 28 vertices:
    # Partition into 5 parts of sizes 6,6,6,5,5
    # Connect parts in a cycle pattern with specific edge rules
    # For simplicity, let's use a different approach:
    # Start with C_5 and blow up each vertex
    
    # Actually, let me use a simpler known valid graph:
    # The Clebsch graph has 16 vertices, not large enough
    # Let's just use a random search from the start, but with a good seed
    
    # Better: use a systematic construction
    # The Andrássy graph And(9) has 27 vertices
    # Let me construct it properly
    
    # For now, let's just do random search starting from n=40 down
    # But with a better baseline to ensure we always have a valid answer
    
    # Generate a valid 26-vertex graph as fallback (Andrásfai graph)
    def andrasfai_graph(n):
        """Generate Andrásfai graph And(k) where n = 3k-1."""
        if n % 3 != 2:
            return None
        k = (n + 1) // 3
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                # Connect i and j if j-i mod n is in {1, k-1, k, 2k-1}
                diff = (j - i) % n
                if diff in {1, k-1, k, 2*k-1}:
                    edges.append([i, j])
        return edges
    
    # And(9) has n=26 vertices
    fallback_edges = andrasfai_graph(26)
    if fallback_edges is not None and validate(26, fallback_edges):
        best_n = 26
        best_edges = fallback_edges
    else:
        # Use complete graph as fallback (will fail validation but ensures we have something)
        best_n = 1
        best_edges = []
    
    # Try to find larger n from 40 down to best_n+1
    for n in range(40, best_n, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Try multiple random strategies
        for attempt in range(500):
            if time.time() > deadline - 2:
                break
                
            # Strategy: Build a maximal triangle-free graph
            # Use a random order and greedily add edges
            edges = []
            adj = [0] * n
            
            # Random permutation of vertices
            order = list(range(n))
            random.shuffle(order)
            
            # For each vertex in order, connect to earlier vertices if no triangle
            for i in range(n):
                u = order[i]
                for j in range(i):
                    v = order[j]
                    # Check if adding (u,v) creates a triangle
                    # Triangle if there exists w connected to both u and v
                    common = adj[u] & adj[v]
                    if common == 0:
                        edges.append([u, v])
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Check if this graph is valid
            if validate(n, edges):
                best_n = n
                best_edges = edges
                found = True
                break
        
        # If we found a valid graph for this n, we can return immediately
        # since we're iterating from high to low
        if found:
            return {'n': n, 'edges': best_edges}
        
        # If we didn't find one for this n, continue to n-1

    return {'n': best_n, 'edges': best_edges}
