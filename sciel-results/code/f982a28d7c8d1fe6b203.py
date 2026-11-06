"""Rewritten program for C4-free graph construction without independent 11-sets."""
import time
import random
import math

# Fixed seed for reproducibility
random.seed(42)

def construct():
    """
    Construct a C4-free graph with no independent 11-set.
    Uses probabilistic construction inspired by Erdős-Spencer style arguments.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_n = 0
    best_edges = None
    
    # Try to find a C4-free graph with alpha < 11
    # Strategy: Use a random graph G(n, p) and then remove edges to eliminate C4s
    # while keeping the independence number low
    
    # The key insight: for C4-free graphs, the number of edges is bounded by O(n^{3/2})
    # We want a graph with enough edges to keep alpha < 11 but no C4s
    
    # Approach: Start with a random graph and iteratively remove one edge from each C4
    # This is expensive, so we use a smarter approach
    
    # Better approach: Use a known construction or random search with verification
    
    # Let's try a direct random construction with C4 avoidance during edge addition
    
    def try_random_construction(n, max_edges=None, time_limit=5.0):
        """
        Try to construct a C4-free graph on n vertices with alpha < 11.
        Uses randomized edge addition with C4 checking.
        """
        t0 = time.time()
        
        # Initialize adjacency as sets
        adj = [set() for _ in range(n)]
        edges = []
        
        # Target number of edges: for C4-free graphs, max edges ~ n*sqrt(n)/2
        # We need enough edges to keep alpha < 11
        # For n=40, we need roughly 40*sqrt(40)/2 ≈ 126 edges
        if max_edges is None:
            max_edges = int(n * math.sqrt(n) / 2)
        
        # Try to add edges randomly while maintaining C4-free property
        # Use a greedy approach: add edges that don't create C4s
        
        # Precompute which edge additions would create a C4
        # An edge (u,v) creates a C4 if there exist a,b such that 
        # a~u, a~v, b~u, b~v (i.e., u and v have two common neighbors)
        
        # For efficiency, track for each pair of vertices how many common neighbors they have
        common_count = [[0] * n for _ in range(n)]
        
        # Initialize common counts
        for u in range(n):
            for v in range(u + 1, n):
                common_count[u][v] = 0
                common_count[v][u] = 0
        
        # Add edges one by one
        # We'll try multiple times with different random orderings
        for attempt in range(100):
            if time.time() - t0 > time_limit:
                break
            
            adj = [set() for _ in range(n)]
            edges = []
            common_count = [[0] * n for _ in range(n)]
            
            # Create a list of all possible edges
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            
            added = 0
            for u, v in all_edges:
                if time.time() - t0 > time_limit:
                    return None
                
                # Check if adding (u,v) creates a C4
                # This happens if common_count[u][v] >= 1 
                # (because after adding, they'd have common_count[u][v] + 1 common neighbors,
                #  but actually we need to check: currently they have common_count[u][v] common neighbors,
                #  adding the edge doesn't change common neighbors, it just adds the edge itself)
                
                # Wait, I need to reconsider. The C4 condition is: no two vertices have two or more
                # common neighbors. Adding edge (u,v) doesn't change the common neighbors of u and v.
                # It changes the common neighbors of other pairs.
                
                # Specifically, adding edge (u,v) means:
                # - For any w that is a neighbor of u (w != v), w and v now have u as a common neighbor
                # - For any w that is a neighbor of v (w != u), w and u now have v as a common neighbor
                
                # So we need to check:
                # 1. For each w in adj[u] (w != v): common_count[w][v] must be 0 before adding
                #    (because after adding, it becomes 1, which is still < 2, so OK)
                #    Actually, if common_count[w][v] is already 1, then after adding it becomes 2 -> C4!
                # 2. For each w in adj[v] (w != u): common_count[w][u] must be 0 before adding
                
                valid = True
                for w in adj[u]:
                    if w != v and common_count[w][v] >= 1:
                        valid = False
                        break
                if valid:
                    for w in adj[v]:
                        if w != u and common_count[w][u] >= 1:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Add the edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
                # Update common counts
                for w in adj[u]:
                    if w != v:
                        common_count[w][v] += 1
                        common_count[v][w] += 1
                for w in adj[v]:
                    if w != u:
                        common_count[w][u] += 1
                        common_count[u][w] += 1
                
                added += 1
                if added >= max_edges:
                    break
            
            # Check if we have enough edges and alpha < 11
            if len(edges) < 20:  # Too few edges, unlikely to have alpha < 11
                continue
            
            # Check C4-free property (should be true by construction, but verify)
            if not _check_no_c4(n, edges):
                continue
            
            # Check independence number
            alpha = _max_independent_set(n, edges, timeout=2.0)
            if alpha < 11:
                return edges
        
        return None
    
    def _check_no_c4(n, edges):
        """Check that no two vertices have two common neighbours."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return False
        return True
    
    def _max_independent_set(n, edges, timeout=2.0):
        """Find maximum independent set using branch and bound with coloring."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        t0 = time.time()
        
        def greedy_color(vertices):
            """Greedy coloring to get upper bound on independence number."""
            color = {}
            for v in vertices:
                used_colors = set()
                for u in adj[v]:
                    if u in color:
                        used_colors.add(color[u])
                c = 0
                while c in used_colors:
                    c += 1
                color[v] = c
            return max(color.values()) + 1 if color else 0
        
        def branch_and_bound(remaining, current_size, best):
            nonlocal t0
            if time.time() - t0 > timeout:
                return best
            if not remaining:
                return best
            if current_size + greedy_color(remaining) <= best:
                return best
            
            # Pick vertex with highest degree in remaining
            v = max(remaining, key=lambda x: len(adj[x] & set(remaining)))
            
            # Include v
            new_remaining = [u for u in remaining if u != v and u not in adj[v]]
            best = branch_and_bound(new_remaining, current_size + 1, best)
            
            # Exclude v
            remaining.remove(v)
            best = branch_and_bound(remaining, current_size, best)
            remaining.append(v)
            
            return best
        
        # Try multiple random orderings for initial greedy solution
        best = 0
        for attempt in range(20):
            if time.time() - t0 > timeout:
                break
            verts = list(range(n))
            random.shuffle(verts)
            current = list(verts)
            ind = []
            while current:
                v = current.pop()
                ind.append(v)
                current = [u for u in current if u not in adj[v]]
            best = max(best, len(ind))
        
        # If we already found an independent set of size >= 11, we can return early
        if best >= 11:
            return best
        
        # Now do branch and bound to find the true maximum
        best = branch_and_bound(list(range(n)), 0, best)
        return best
    
    # Try different n values from high to low
    for n in range(43, 29, -1):
        if time.time() > deadline:
            break
        
        # Try to construct a valid graph on n vertices
        result = try_random_construction(n, time_limit=10.0)
        if result is not None:
            # Verify
            if _check_no_c4(n, result) and _max_independent_set(n, result, timeout=3.0) < 11:
                best_n = n
                best_edges = result
                break
    
    # If we didn't find anything good, use a fallback
    if best_n == 0:
        # Fallback: 30 vertices with 10 disjoint triangles
        n = 30
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best_n = n
        best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}
