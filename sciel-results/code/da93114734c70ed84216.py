"""Search-based bootstrap for the R(3,10) triangle-free graph task."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 125.0  # 150s limit, keep 25s margin

    best_n = 0
    best_edges = []

    def is_valid(n, edges, max_check_time=None):
        """Check triangle-free and independence number < 10.
        Returns True if valid (alpha < 10 and no triangle)."""
        if max_check_time is not None and time.time() > max_check_time:
            return False
        
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Triangle check
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        
        # Independence number check: ensure no independent set of size 10.
        # Use branch and bound to find max independent set size.
        # If max independent set >= 10, invalid.
        
        def find_max_independent_set(n, adj, timeout_check=None):
            """Find size of maximum independent set via branch and bound."""
            # Use a greedy coloring bound on the complement graph for pruning
            # But simpler: use recursive search with upper bound.
            # For n up to 40, this can be expensive.
            
            # Alternative: check if there exists an independent set of size 10.
            # We can do this with a recursive search that stops as soon as we find one.
            
            def exists_independent_set_of_size(k, candidates):
                """Check if there's an independent set of size k among candidates."""
                if len(candidates) < k:
                    return False
                if k == 0:
                    return True
                if k == 1:
                    return len(candidates) > 0
                
                # Upper bound: can we pick k from candidates such that they're independent?
                # Simple bound: if len(candidates) < k, no.
                
                # Choose a vertex to branch on
                # Heuristic: pick vertex with highest degree in the candidate subgraph (in complement, that's min degree in original)
                # Actually, for independent set, we want vertices that don't conflict.
                # Let's just pick the first one for simplicity, or use a heuristic.
                
                # Pick a pivot: vertex with max degree in the induced subgraph on candidates
                # High degree means fewer choices, prune earlier
                pivot_idx = -1
                pivot_deg = -1
                for idx, v in enumerate(candidates):
                    deg = sum(1 for u in candidates if u != v and adj[v][u])
                    if deg > pivot_deg:
                        pivot_deg = deg
                        pivot_idx = idx
                
                pivot = candidates[pivot_idx]
                
                # Option 1: include pivot
                # New candidates: those not adjacent to pivot, excluding pivot itself
                new_candidates = [v for v in candidates if v != pivot and not adj[pivot][v]]
                if exists_independent_set_of_size(k - 1, new_candidates):
                    return True
                
                # Option 2: exclude pivot
                new_candidates = [v for v in candidates if v != pivot]
                if exists_independent_set_of_size(k, new_candidates):
                    return True
                
                return False
            
            return exists_independent_set_of_size(10, list(range(n)))
        
        # Check if independent set of size 10 exists
        has_large_independent = find_max_independent_set(n, adj)
        return not has_large_independent

    def greedy_construct(target_n, edge_prob=0.5):
        """Build a triangle-free graph on target_n vertices with alpha < 10."""
        n = target_n
        adj = [[False]*n for _ in range(n)]
        
        # Try to add edges to make the graph connected and with small alpha
        # Use a random order for adding edges
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if random.random() >= edge_prob:
                continue
            # Check if adding (u,v) creates a triangle
            creates_triangle = False
            for k in range(n):
                if k != u and k != v and adj[u][k] and adj[v][k]:
                    creates_triangle = True
                    break
            if not creates_triangle:
                adj[u][v] = True
                adj[v][u] = True
        
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        
        return n, edges

    def remove_triangles_from_graph(n, adj):
        """Remove edges to destroy all triangles. Returns new adj matrix."""
        adj = [row[:] for row in adj]
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        # Check for triangles involving edge (i,j)
                        for k in range(j+1, n):
                            if adj[i][k] and adj[j][k]:
                                # Remove one of the edges to break the triangle
                                # Remove the edge with highest "cost" (most triangles it participates in)
                                # For simplicity, remove (i,k) or (j,k)
                                # Let's remove (i,k)
                                adj[i][k] = False
                                adj[k][i] = False
                                changed = True
                                break
        return adj

    def random_regular_graph_complement(n, degree):
        """Generate complement of a random regular graph using random perfect matchings."""
        if n % 2 != 0:
            return None
        
        adj = [[False]*n for _ in range(n)]
        
        # Generate degree/2 random perfect matchings
        num_matchings = degree // 2
        for _ in range(num_matchings):
            # Generate a random perfect matching
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(0, n, 2):
                u, v = vertices[i], vertices[i+1]
                adj[u][v] = True
                adj[v][u] = True
        
        # Return the complement
        comp_adj = [[not adj[i][j] for j in range(n)] for i in range(n)]
        for i in range(n):
            comp_adj[i][i] = False
        
        return comp_adj

    def greedy_remove_triangles(n, comp_adj, max_iterations=100):
        """Remove edges from complement graph to destroy triangles, guided by independence heuristic."""
        adj = [row[:] for row in comp_adj]
        
        for _ in range(max_iterations):
            if time.time() > DEADLINE:
                break
            
            # Find all triangles
            triangles = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        for k in range(j+1, n):
                            if adj[i][k] and adj[j][k]:
                                triangles.append((i, j, k))
            
            if not triangles:
                break
            
            # For each triangle, decide which edge to remove
            # Heuristic: remove edge that most reduces independence number
            # Approximate by removing edge between vertices with highest degrees
            for i, j, k in triangles:
                if not (adj[i][j] and adj[i][k] and adj[j][k]):
                    continue
                
                # Compute degrees
                deg_i = sum(adj[i])
                deg_j = sum(adj[j])
                deg_k = sum(adj[k])
                
                # Remove edge with highest sum of endpoint degrees
                options = [
                    (i, j, deg_i + deg_j),
                    (i, k, deg_i + deg_k),
                    (j, k, deg_j + deg_k)
                ]
                options.sort(key=lambda x: -x[2])
                
                u, v, _ = options[0]
                adj[u][v] = False
                adj[v][u] = False
            
            # Check if we still have triangles
            has_triangle = False
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        for k in range(j+1, n):
                            if adj[i][k] and adj[j][k]:
                                has_triangle = True
                                break
                        if has_triangle:
                            break
                if has_triangle:
                    break
            
            if not has_triangle:
                break
        
        return adj

    # Phase 1: Try known constructions
    # Andrasfai graph of order 26 (current seed)
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    if is_valid(n, edges):
        best_n = n
        best_edges = edges

    # Phase 2: Greedy construction for various n
    for n in range(27, 41):
        if time.time() > DEADLINE:
            break
        
        for edge_prob in [0.3, 0.5, 0.7]:
            if time.time() > DEADLINE:
                break
            
            for attempt in range(3):
                if time.time() > DEADLINE:
                    break
                
                _, edges = greedy_construct(n, edge_prob)
                if is_valid(n, edges):
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break

    # Phase 3: Complement of random regular graph approach
    for n in range(28, 41):
        if time.time() > DEADLINE:
            break
        
        if n % 2 != 0:
            continue
        
        for degree in [4, 6, 8]:
            if time.time() > DEADLINE:
                break
            
            for attempt in range(5):
                if time.time() > DEADLINE:
                    break
                
                comp_adj = random_regular_graph_complement(n, degree)
                if comp_adj is None:
                    continue
                
                # Remove triangles from complement
                final_adj = greedy_remove_triangles(n, comp_adj, max_iterations=50)
                
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if final_adj[i][j]:
                            edges.append([i, j])
                
                if is_valid(n, edges):
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break

    # Phase 4: More aggressive greedy with higher edge probability
    for n in range(30, 41):
        if time.time() > DEADLINE:
            break
        
        for edge_prob in [0.8, 0.9]:
            if time.time() > DEADLINE:
                break
            
            for attempt in range(10):
                if time.time() > DEADLINE:
                    break
                
                _, edges = greedy_construct(n, edge_prob)
                if is_valid(n, edges):
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    break

    # Final validation
    if best_n == 0:
        # Fallback to the known valid graph
        n = 26
        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
