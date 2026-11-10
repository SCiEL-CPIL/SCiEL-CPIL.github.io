"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_k4_free(adj):
    """Check if graph (adjacency matrix) is K4-free."""
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_size(adj, deadline):
    """Find the largest triangle-free induced subgraph size using branch and bound."""
    n = len(adj)
    
    def is_triangle_free(vertices):
        """Check if the given set of vertices induces a triangle-free subgraph."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                for k in range(j+1, len(vertices)):
                    u, v, w = vertices[i], vertices[j], vertices[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        return False
        return True
    
    # Branch and bound: try to find if there's a triangle-free subset of size > best
    best = 0
    # Use greedy upper bound: for a set S, if it contains a triangle, we can't use all of it
    
    def upper_bound(candidates):
        """Upper bound on how many more vertices we can add."""
        return len(candidates)
    
    # Simple approach: try all subsets of size s from high to low
    # For n=18, this is feasible with pruning
    
    # Better: use the fact that a triangle-free set can be built greedily
    # Start with all vertices, remove vertices that are in triangles
    
    def find_max_tf(adj, n, deadline):
        best_size = 0
        best_set = set()
        
        # Try sizes from n down to 1
        # For each size, check if any subset of that size is triangle-free
        # Use backtracking with pruning
        
        def backtrack(available, current, size_needed):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return False
            if len(current) == size_needed:
                # Check if current is triangle-free
                verts = list(current)
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        for k in range(j+1, len(verts)):
                            if adj[verts[i]][verts[j]] and adj[verts[j]][verts[k]] and adj[verts[i]][verts[k]]:
                                return False
                best_size = size_needed
                best_set = set(current)
                return True
            if len(available) < size_needed - len(current):
                return False
            
            # Pruning: if current already has a triangle, stop
            verts = list(current)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    for k in range(j+1, len(verts)):
                        if adj[verts[i]][verts[j]] and adj[verts[j]][verts[k]] and adj[verts[i]][verts[k]]:
                            return False
            
            for i, v in enumerate(available):
                if time.time() > deadline:
                    return False
                # Check if adding v creates a triangle with any pair in current
                valid = True
                for a in current:
                    for b in current:
                        if a != b and adj[v][a] and adj[v][b] and adj[a][b]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_current = current | {v}
                    new_available = set(available[i+1:])
                    if backtrack(new_available, new_current, size_needed):
                        return True
            return False
        
        for size in range(n, 0, -1):
            if backtrack(set(range(n)), set(), size):
                return best_size
        return best_size
    
    return find_max_tf(adj, n, deadline)

def _compute_score(adj):
    """Compute the largest triangle-free induced subgraph size."""
    n = len(adj)
    # Use a more efficient approach for n=18
    # Try all subsets of size s from high to low
    # For s >= 10, the number of subsets is manageable with pruning
    
    def is_tf(vertices):
        """Check if vertices induce a triangle-free subgraph."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                for k in range(j+1, len(vertices)):
                    u, v, w = vertices[i], vertices[j], vertices[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        return False
        return True
    
    # Check sizes from n down to 1
    for s in range(n, 0, -1):
        # For each combination of s vertices, check if triangle-free
        # Use early termination
        found = False
        for combo in itertools.combinations(range(n), s):
            if is_tf(combo):
                return s
        # If no triangle-free subset of size s, continue to s-1
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with K6,6,6
    n = 18
    parts = [set(range(0, 6)), set(range(6, 12)), set(range(12, 18))]
    
    # Build adjacency matrix for K6,6,6
    adj = [[False]*n for _ in range(n)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = True
                    adj[b][a] = True
    
    # Verify K4-free
    assert _is_k4_free(adj)
    
    # Compute score
    base_score = _compute_score(adj)
    best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if adj[a][b]]
    best_score = base_score
    
    # Try local search: remove random edges and see if score improves
    # Note: removing edges can only increase or maintain the largest TF subset
    # So this won't help. We need a different structure.
    
    # Try alternative constructions
    # Idea: Use a non-tripartite K4-free graph
    
    # Construction: Take vertices 0-17, create a graph where:
    # - Partition into 6 groups of 3
    # - Between groups, use a specific pattern to avoid K4s
    # This is complex; let's try a simpler approach
    
    # Try: complete graph minus edges to make it K4-free
    # Start with K18, remove edges to break all K4s
    # This is like finding a K4-free graph with many triangles spread out
    
    # Alternative: use the line graph of K6 or other known K4-free graphs
    # Line graph of K6 has 15 vertices. Not enough.
    
    # Let's try: 3 groups of 6, but with a different inter-group pattern
    # Instead of complete tripartite, use a "blow-up" of a triangle-free base
    # Wait, we need triangles to exist for the score to be small
    
    # Key insight: We need triangles in every large subset.
    # In K6,6,6, triangles require all 3 parts.
    # If we could make triangles appear in 2-part subsets, we'd do better.
    # But adding intra-part edges creates K4s.
    
    # What if we use a different partition? E.g., 2 groups of 9?
    # Complete bipartite K9,9 is triangle-free, so score = 18. Terrible.
    
    # What about: 3 groups of 6, but between groups use a regular bipartite graph
    # that is not complete? This reduces edges and might reduce triangles.
    # But we WANT more triangles to reduce the max TF subset.
    
    # I think K6,6,6 might be optimal. Let's verify by trying a few alternatives.
    
    # Alternative 1: K6,6,6 with some edges removed (worse or equal)
    # Alternative 2: Different structure
    
    # Let's try: take the complement of a 3-regular graph on 18 vertices
    # The complement has degree 14. Check if K4-free.
    # A K4 in complement = independent set of size 4 in original.
    # A 3-regular graph on 18 vertices: does it have an independent set of size 4?
    # By Ramsey R(3,4)=9, any graph on 9+ vertices has a triangle or independent set of 4.
    # So a 3-regular graph on 18 vertices might or might not have an independent set of 4.
    
    # Let's try a specific 3-regular graph: the 3-prism graph or similar
    # Actually, let's just try random 3-regular graphs
    
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random 3-regular graph on 18 vertices
        # Using configuration model
        while True:
            if time.time() > deadline:
                break
            # Create stubs
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            # Pair them up
            adj2 = [[False]*n for _ in range(n)]
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or (u,v) in seen or (v,u) in seen:
                    valid = False
                    break
                adj2[u][v] = True
                adj2[v][u] = True
                seen.add((u,v))
            if valid:
                break
        else:
            continue
        
        # Check if adj2 is simple (no multi-edges, no self-loops) - already ensured
        # Build complement
        comp = [[not adj2[i][j] and i != j for j in range(n)] for i in range(n)]
        
        # Check if complement is K4-free
        if not _is_k4_free(comp):
            continue
        
        # Compute score
        score = _compute_score(comp)
        if score < best_score:
            best_score = score
            best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if comp[a][b]]
    
    # Try more: random K4-free graphs via edge addition
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Start with empty graph, add random edges while maintaining K4-free
        adj3 = [[False]*n for _ in range(n)]
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            # Try adding edge (u,v)
            adj3[u][v] = True
            adj3[v][u] = True
            if not _is_k4_free(adj3):
                adj3[u][v] = False
                adj3[v][u] = False
        
        if not _is_k4_free(adj3):
            continue
        
        score = _compute_score(adj3)
        if score < best_score:
            best_score = score
            best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if adj3[a][b]]
    
    return best_edges
# EVOLVE-BLOCK-END
