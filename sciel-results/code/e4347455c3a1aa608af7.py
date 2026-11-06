import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145
    
    # Construct Hoffman-Singleton Graph
    # 50 vertices, 7-regular, girth 5
    # Construction:
    # Vertices 0..49
    # 0 is connected to 1..7? No.
    # Standard construction:
    # Vertices are pairs (i, j) where i in 0..4, j in 0..9? No.
    # Let's use the known adjacency list or generate it.
    
    # A simpler C4-free graph: Petersen graph is small.
    # Let's try to build a graph on 39 vertices from scratch using a greedy method?
    # Or use the HS graph.
    
    # HS Graph Adjacency:
    # One construction:
    # V = {0, ..., 49}
    # E = {(i, i+1) mod 50} U {(i, i+5) mod 50} ... this is not HS.
    
    # Let's use a known edge list for HS graph if possible, or generate.
    # Since I don't have the edge list hardcoded, I will generate a random C4-free graph and try to optimize.
    # However, random generation of C4-free graphs with specific alpha is hard.
    
    # Alternative: Use the 10-triangle base (n=30) and try to add vertices.
    # We established that connecting triangles with matchings creates C4s.
    # What if we connect them with a single edge?
    # If we connect triangle A and B with a single edge (a0, b0).
    # Common neighbors of a0, a1: a2. (b0 is not neighbor of a1).
    # Common neighbors of a0, b1: 
    #   N(a0) = {a1, a2, b0}
    #   N(b1) = {b0, b2, a0? No, a0-b0 is edge. b1 is not connected to a0 directly unless it's the edge.}
    #   If edge is (a0, b0), then a0 ~ b0.
    #   N(a0) = {a1, a2, b0}
    #   N(b1) = {b0, b2} (within B)
    #   Common: {b0}. Count 1.
    #   So single edge connections are safe!
    
    # If we connect 10 triangles in a path or cycle using single edges, we get a C4-free graph.
    # n = 30.
    # Can we add more vertices?
    # We can add more triangles and connect them with single edges.
    # If we have k triangles, n = 3k.
    # Independence number:
    # If we pick one vertex from each triangle, can we form an independent set?
    # If the triangles are connected by single edges, the "inter-triangle" edges might prevent independence.
    # If we pick a0 from A and b0 from B, and there is an edge (a0, b0), they are NOT independent.
    # So we must pick vertices that are not connected by inter-triangle edges.
    
    # If we connect the triangles in a chain: T1-T2-...-Tk.
    # Edge between Ti and Ti+1 is (u_i, v_{i+1}).
    # To form an independent set, we pick 1 vertex from each Ti.
    # Let picked vertex in Ti be x_i.
    # If x_i = u_i and x_{i+1} = v_{i+1}, they are adjacent.
    # So we cannot pick both endpoints of the connecting edge.
    # This is like an independent set on a path of "states".
    # For each triangle, we have 3 choices.
    # The constraint is local.
    # The maximum independent set size will be roughly k (one per triangle) minus some penalties.
    # If we have 13 triangles, n=39.
    # Can we have an independent set of size 11?
    # If we pick vertices to avoid the connecting edges, we can likely pick 1 from each triangle.
    # So alpha would be 13.
    # We need alpha < 11.
    # So 13 triangles is too many.
    # 10 triangles (n=30) has alpha=10 (if we pick carefully? No, if we pick the non-connected vertices, we can pick 1 from each. So alpha=10).
    # If we add 3 more triangles (total 13, n=39), alpha will be at least 11?
    # If we can force alpha to be 10 by connecting them in a way that breaks independence?
    # If we connect T1 and T2, T2 and T3, etc.
    # If we pick x1 from T1, we cannot pick v2 from T2 if x1=u1.
    # But we can pick another vertex from T2.
    # So we can still pick 1 from each triangle.
    # Thus alpha = k.
    # To have alpha < 11, we need k <= 10.
    # So n=30 is the max for this construction.
    
    # We need a denser connection or a different base.
    
    # Let's try to find a C4-free graph on 39 vertices using a random search with a strong C4-free constraint.
    # We will start with an empty graph and add edges randomly, checking C4-free.
    # Then we will try to minimize alpha by adding more edges?
    # No, adding edges reduces alpha.
    # So:
    # 1. Generate a random C4-free graph on 39 vertices.
    # 2. Check if alpha < 11.
    # 3. If not, add random edges (while maintaining C4-free) to reduce alpha.
    
    def is_c4_free(n, adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        return False
        return True

    def max_independent_set_size(n, adj, limit=11):
        # Returns True if alpha >= limit
        # Uses branch and bound
        best = 0
        # Order vertices by degree
        order = sorted(range(n), key=lambda x: len(adj[x]))
        
        def backtrack(idx, current_size, chosen):
            nonlocal best
            if current_size >= limit:
                return True
            if idx == n:
                return False
            # Pruning
            if current_size + (n - idx) <= best:
                return False
            
            v = order[idx]
            # Option 1: Include v
            if all(v not in adj[u] for u in chosen):
                if backtrack(idx + 1, current_size + 1, chosen + [v]):
                    return True
            # Option 2: Exclude v
            if backtrack(idx + 1, current_size, chosen):
                return True
            return False
        
        return backtrack(0, 0, [])

    # Random search
    n_target = 39
    best_graph = None
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
            
        # Start with a random sparse C4-free graph
        adj = [set() for _ in range(n_target)]
        
        # Add edges greedily
        random.seed(attempt)
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if random.random() < 0.2: # Try to add edge
                    # Check if adding (u,v) creates C4
                    # Common neighbors of u and v must be < 2 after adding?
                    # No, adding (u,v) makes u and v adjacent.
                    # We need to check if u and v have >= 2 common neighbors ALREADY?
                    # No, if they have 2 common neighbors, they already have a C4.
                    # So we must ensure that before adding (u,v), len(adj[u] & adj[v]) < 2.
                    # Actually, if they have 2 common neighbors, the graph is already not C4-free.
                    # So we just need to check that the graph remains C4-free.
                    # Adding an edge (u,v) doesn't create new common neighbors for other pairs.
                    # It only affects the pair (u,v).
                    # If (u,v) is added, u and v become adjacent.
                    # Do they have 2 common neighbors?
                    # If they had 2 common neighbors before, the graph was invalid.
                    # So if the graph is C4-free so far, adding (u,v) is safe IF u and v have < 2 common neighbors.
                    # Wait. If u and v have 2 common neighbors w1, w2, then u-w1-v-w2-u is a C4.
                    # So we MUST NOT add (u,v) if len(adj[u] & adj[v]) >= 2.
                    
                    if len(adj[u] & adj[v]) < 2:
                        adj[u].add(v)
                        adj[v].add(u)
        
        # Now we have a C4-free graph.
        # Check alpha
        if not max_independent_set_size(n_target, adj, 11):
            # Alpha < 11
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n_target, 'edges': edges}
        
        # If alpha >= 11, try to add more edges to reduce alpha
        # We can try to add edges between non-adjacent vertices that don't create C4s
        improved = False
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u == v or v in adj[u]:
                continue
            if len(adj[u] & adj[v]) < 2:
                adj[u].add(v)
                adj[v].add(u)
                improved = True
        
        if improved:
            if not max_independent_set_size(n_target, adj, 11):
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n_target, 'edges': edges}

    # Fallback: Return the 30-vertex graph
    best_edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        best_edges += [[a,b],[a,c],[b,c]]
    return {'n': 30, 'edges': best_edges}
