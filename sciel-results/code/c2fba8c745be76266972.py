"""Find maximum-edge triangle-free non-3-colorable graph on 16 vertices."""
import time
import random

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 16
    rng = random.Random(42)
    
    # All possible edges as frozensets for quick lookup
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Build the Clebsch graph (16 vertices, 40 edges, triangle-free, 3-colorable)
    # The Clebsch graph is the folded 5-cube. It's a strongly regular graph (16,5,0,2).
    # Vertices are 4-bit strings; two vertices are adjacent if they differ in exactly 2 bits.
    clebsch_edges = []
    for i in range(16):
        for j in range(i+1, 16):
            diff = i ^ j
            # Count bits
            bit_count = bin(diff).count('1')
            if bit_count == 2:
                clebsch_edges.append([i, j])
    
    # Verify: should be 40 edges
    # assert len(clebsch_edges) == 40
    
    def edge_set(edges):
        s = set()
        for e in edges:
            s.add(frozenset((e[0], e[1])))
        return s
    
    def is_triangle_free(adj_set):
        # For each edge (u,v), check no common neighbor
        for e in adj_set:
            u, v = tuple(e)
            # Find neighbors of u
            for e2 in adj_set:
                if u in e2 and v not in e2:
                    w = u ^ v ^ tuple(e2)[0] if tuple(e2)[0] != u else tuple(e2)[1]
                    # Actually let's just check if (v,w) is in adj_set
                    if v in e2:
                        continue
                    w = next(iter(e2 - {u}))
                    if frozenset((v, w)) in adj_set:
                        return False
        return True
    
    def is_triangle_free_fast(adj_set, n):
        """Check triangle-free using bitmasks."""
        # Build adjacency bitmask
        adj_mask = [0] * n
        for e in adj_set:
            u, v = tuple(e)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    # Check if u and v share a common neighbor
                    if adj_mask[u] & adj_mask[v]:
                        return False
        return True
    
    def is_3_colorable(edges, deadline):
        """Check if graph is 3-colorable using backtracking."""
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Sort vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj_list[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        colors = [-1] * n
        color_count = [0, 0, 0]
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = order[idx]
            # Determine available colors
            used = set()
            for nb in adj_list[v]:
                if colors[nb] >= 0:
                    used.add(colors[nb])
            
            # Try colors in order of least constraining first
            for c in range(3):
                if c not in used:
                    colors[v] = c
                    color_count[c] += 1
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
                    color_count[c] -= 1
            return False
        
        return backtrack(0)
    
    def count_edges(edges):
        return len(edges)
    
    # Start from Clebsch graph
    current_edges = clebsch_edges[:]
    current_adj = edge_set(current_edges)
    
    # We need to find a non-3-colorable triangle-free subgraph with max edges.
    # Strategy: Use simulated annealing / local search.
    # State: a set of edges (triangle-free).
    # Move: add an edge, or remove an edge, or swap.
    # Objective: maximize edge count among non-3-colorable triangle-free graphs.
    # 
    # Actually, since we want max edges, and the Clebsch graph is 3-colorable,
    # we need to find a dense triangle-free graph that is NOT 3-colorable.
    #
    # Key insight: The complement of a triangle-free graph on 16 vertices...
    # Actually, let's think about this differently.
    #
    # A triangle-free graph on 16 vertices with max edges: by Turan's theorem,
    # the max is floor(16^2/4) = 64 (complete bipartite K8,8).
    # K8,8 is bipartite, hence 2-colorable, hence 3-colorable.
    #
    # So we need a triangle-free graph that is NOT bipartite (i.e., has odd cycles)
    # and is not 3-colorable.
    #
    # The Grötzsch graph (Mycielski of C5) has 11 vertices and 20 edges, is
    # triangle-free and not 3-colorable. Chromatic number 4.
    #
    # We want to extend this to 16 vertices with more edges.
    #
    # Let's try a different approach: start from a known 4-chromatic triangle-free
    # graph on 16 vertices and try to add edges.
    #
    # The Clebsch graph is 3-colorable. What about the Schrijver graph?
    # Or we can use the following: take the Grötzsch graph on 11 vertices,
    # add 5 more vertices, and connect them to maximize edges while maintaining
    # triangle-free and 4-chromatic.
    #
    # Actually, let's use a more systematic approach:
    # 1. Start with the Clebsch graph (40 edges, 3-colorable).
    # 2. Remove edges to make it non-3-colorable (minimize removals).
    # 3. Then try to add back edges that don't create triangles or make it 3-colorable.
    #
    # Or better: use a local search that directly optimizes for non-3-colorability.
    
    def is_non_3_colorable(edges):
        return not is_3_colorable(edges, deadline)
    
    # Let's use a different strategy:
    # The maximum number of edges in a triangle-free graph on 16 vertices is 64 (K8,8).
    # We want the densest triangle-free graph that is not 3-colorable.
    #
    # This is equivalent to: minimize the number of edges to remove from a
    # triangle-free graph to make it 3-colorable, and we want the graph where
    # this minimum is maximized.
    #
    # Let's try a randomized local search:
    # 1. Start with a random triangle-free graph (e.g., random subgraph of Clebsch).
    # 2. Try to add random edges that keep it triangle-free.
    # 3. If the graph becomes 3-colorable, remove a random edge that was critical.
    # 4. Track the best non-3-colorable graph found.
    
    best_edges = None
    best_count = 0
    
    def evaluate(edges):
        """Return (is_valid, edge_count) where valid means triangle-free and non-3-colorable."""
        if not is_triangle_free_fast(edge_set(edges), n):
            return False, 0
        if is_3_colorable(edges, deadline):
            return False, len(edges)
        return True, len(edges)
    
    # Start with the Clebsch graph
    current = clebsch_edges[:]
    current_adj = edge_set(current)
    
    # The Clebsch graph is 3-colorable, so we need to modify it.
    # Let's try removing edges to break 3-colorability.
    
    # First, find a minimal edge set whose removal makes the Clebsch graph non-3-colorable.
    # This is hard, so let's use a heuristic.
    
    # Alternative: start from the Grötzsch graph and extend it.
    
    def groetzsch_edges():
        """Return edges of the Grötzsch graph (11 vertices, 20 edges)."""
        edges = []
        # C5: 0-1-2-3-4-0
        for i in range(5):
            edges.append([i, (i+1)%5])
        # Mycielski construction
        # For each vertex i in C5, add vertex 5+i
        # For each edge (i, j) in C5, add edges (5+i, j) and (5+j, i)
        for i in range(5):
            j = (i+1)%5
            edges.append([5+i, j])
            edges.append([5+j, i])
        # Add vertex 10 connected to all 5,6,7,8,9
        for i in range(5):
            edges.append([10, 5+i])
        return edges
    
    # Start with Grötzsch graph on vertices 0-10, vertices 11-15 isolated
    current = groetzsch_edges()
    current_adj = edge_set(current)
    
    # Verify it's triangle-free and non-3-colorable
    assert is_triangle_free_fast(current_adj, n), "Grötzsch graph should be triangle-free"
    assert not is_3_colorable(current, deadline), "Grötzsch graph should not be 3-colorable"
    
    best_edges = current[:]
    best_count = len(current)  # 20 edges
    
    # Now try to add edges to increase the count while maintaining triangle-free and non-3-colorable
    # Use a greedy approach with backtracking
    
    def try_add_edges(edges, adj_set, max_time):
        """Try to add edges greedily."""
        if time.time() > max_time:
            return edges, adj_set
        
        current_edges = edges[:]
        current_adj = adj_set.copy()
        
        # Candidate edges not in current graph
        candidates = [e for e in all_edges if frozenset(e) not in current_adj]
        
        # Sort candidates by some heuristic (e.g., between high-degree vertices)
        # Compute degrees
        deg = [0] * n
        for e in current_adj:
            u, v = tuple(e)
            deg[u] += 1
            deg[v] += 1
        
        # Try adding edges in order of potential (high degree endpoints)
        candidates.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
        
        for u, v in candidates:
            if time.time() > max_time:
                break
            
            # Check if adding (u,v) creates a triangle
            if is_triangle_free_fast(current_adj | {frozenset((u,v))}, n):
                new_edges = current_edges + [[u, v]]
                new_adj = current_adj | {frozenset((u,v))}
                
                # Check if still non-3-colorable
                if not is_3_colorable(new_edges, deadline):
                    current_edges = new_edges
                    current_adj = new_adj
        
        return current_edges, current_adj
    
    # Iteratively try to add edges
    for _ in range(100):  # up to 100 iterations
        if time.time() > deadline:
            break
        
        current, current_adj = try_add_edges(current, current_adj, deadline)
        
        if len(current) > best_count:
            best_edges = current[:]
            best_count = len(current)
        
        # If we can't add more edges, try a different approach
        # Remove a random edge and try again
        if len(current) == best_count and best_count > 0:
            if current:
                remove_idx = rng.randrange(len(current))
                current = current[:remove_idx] + current[remove_idx+1:]
                current_adj = edge_set(current)
    
    # Also try starting from Clebsch graph with edge removals
    # to find a non-3-colorable subgraph
    
    def clebsch_with_removals():
        """Try to find a non-3-colorable subgraph of Clebsch with max edges."""
        if time.time() > deadline:
            return None
        
        # Start with all Clebsch edges
        edges = clebsch_edges[:]
        adj = edge_set(edges)
        
        # The Clebsch graph is 3-colorable. We need to remove edges to make it non-3-colorable.
        # This is tricky. Let's try random removals and check.
        
        best = None
        best_count = 0
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Start with a random subset of Clebsch edges
            num_edges_to_keep = rng.randint(20, 40)
            random_edges = rng.sample(clebsch_edges, min(num_edges_to_keep, len(clebsch_edges)))
            
            # Check if triangle-free (should be, since it's a subgraph of Clebsch)
            adj_set = edge_set(random_edges)
            if not is_triangle_free_fast(adj_set, n):
                continue
            
            # Check if non-3-colorable
            if not is_3_colorable(random_edges, deadline):
                if len(random_edges) > best_count:
                    best = random_edges[:]
                    best_count = len(random_edges)
        
        return best
    
    clebsch_result = clebsch_with_removals()
    if clebsch_result and len(clebsch_result) > best_count:
        best_edges = clebsch_result
        best_count = len(clebsch_result)
    
    # Final verification
    if best_edges:
        adj_final = edge_set(best_edges)
        assert is_triangle_free_fast(adj_final, n), "Result should be triangle-free"
        assert not is_3_colorable(best_edges, deadline), "Result should not be 3-colorable"
    
    return best_edges if best_edges else clebsch_edges
