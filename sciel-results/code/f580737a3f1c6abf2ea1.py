import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Define Chvatal Graph (12 vertices, triangle-free, alpha=4)
    # Standard Chvatal graph edges
    chvatal_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (10, 11), (11, 0), (0, 6), (6, 1), (1, 7), (7, 2),
        (2, 8), (8, 3), (3, 9), (9, 4), (4, 5), (5, 10)
    ]
    
    base_n = 12
    base_edges = set(chvatal_edges)
    
    # 2. Extend to 13 vertices
    # Add vertex 12, connect to 0
    base_edges_13 = set(base_edges)
    base_edges_13.add((12, 0))
    
    # 3. Extend to 14 vertices
    # Add vertex 13, connect to 1
    base_edges_14 = set(base_edges_13)
    base_edges_14.add((13, 1))
    
    candidates = []
    
    # Candidate 1: Blow up 14-vertex base to n=40
    # 12 parts of size 3, 2 parts of size 2
    # Total 12*3 + 2*2 = 40
    # Map base vertices to sizes
    sizes_40 = [3] * 12 + [2, 2]
    
    # Candidate 2: Blow up 13-vertex base to n=39
    sizes_39 = [3] * 13
    
    # Candidate 3: Blow up 12-vertex base to n=36
    sizes_36 = [3] * 12
    
    def build_blowup(base_edge_set, base_n, sizes):
        n = sum(sizes)
        # Map base vertex to range of new vertices
        offsets = [0] * (base_n + 1)
        for i in range(base_n):
            offsets[i + 1] = offsets[i] + sizes[i]
        
        edges = []
        # Internal edges (cliques within parts)
        for i in range(base_n):
            start = offsets[i]
            end = offsets[i + 1]
            for u in range(start, end):
                for v in range(u + 1, end):
                    edges.append([u, v])
        
        # External edges (complete bipartite between parts if edge in base)
        for (u, v) in base_edge_set:
            if u == v: continue
            start_u = offsets[u]
            end_u = offsets[u + 1]
            start_v = offsets[v]
            end_v = offsets[v + 1]
            for i in range(start_u, end_u):
                for j in range(start_v, end_v):
                    edges.append([i, j])
        
        return {'n': n, 'edges': edges}
    
    def is_valid(graph):
        n = graph['n']
        edges = graph['edges']
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check K4
        # For each edge (u,v), check if they have 2 common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Count bits
                    if bin(common).count('1') >= 2:
                        return False
        
        # Check Independent Set of size 6
        # Use bitmask backtracking or simple check for n <= 40
        # Since n is small, we can check all subsets of size 6
        # But 40 choose 6 is ~3.8M. 
        # Optimization: If we find one, return False.
        
        # Faster check: 
        # Greedy or Branch and Bound for alpha >= 6
        # Or simply: 
        # For each vertex v, its non-neighbors must not contain an independent set of size 5.
        
        # Let's use a simple recursive check for alpha >= 6
        def has_independent_set(vertices, k):
            if k == 0:
                return True
            if not vertices:
                return False
            # Pick a vertex
            v = vertices[0]
            rest = vertices[1:]
            # Option 1: Include v. Then we can only include non-neighbors of v from rest.
            non_neighbors = [u for u in rest if not (adj[v] >> u) & 1]
            if has_independent_set(non_neighbors, k - 1):
                return True
            # Option 2: Exclude v
            if has_independent_set(rest, k):
                return True
            return False
        
        if has_independent_set(list(range(n)), 6):
            return False
            
        return True

    # Try candidates in order of n
    test_cases = [
        (base_edges_14, 14, sizes_40),
        (base_edges_13, 13, sizes_39),
        (base_edges, 12, sizes_36)
    ]
    
    best_graph = None
    for base_edges, base_n, sizes in test_cases:
        if time.time() > deadline:
            break
        g = build_blowup(base_edges, base_n, sizes)
        if is_valid(g):
            best_graph = g
            break
        
    if best_graph is None:
        # Fallback to seed
        squares = {x * x % 17 for x in range(1, 17)}
        best_graph = {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
        
    return best_graph
