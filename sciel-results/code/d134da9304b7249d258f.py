"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Build adjacency for base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free means bipartite. Path is bipartite with even/odd split.
    # We can only add edges between even and odd vertices.
    # Candidate edges: all (even, odd) pairs not already in base_edges
    even_vertices = [v for v in range(n) if v % 2 == 0]
    odd_vertices = [v for v in range(n) if v % 2 == 1]
    
    candidate_edges = []
    for u in even_vertices:
        for v in odd_vertices:
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidate_edges.append((u, v))
    
    # BFS to compute diameter
    def compute_diameter(edge_set):
        adj_local = [set() for _ in range(n)]
        for u, v in edge_set:
            adj_local[u].add(v)
            adj_local[v].add(u)
        # Check connectivity and compute diameter
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for i in range(len(queue)):
                u = queue[i]
                for w in adj_local[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        queue.append(w)
            if -1 in dist:
                return float('inf')  # Not connected
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    # Try to find minimum number of edges
    # Start with a good heuristic solution, then try to improve
    
    # Heuristic: hub-and-spoke with vertex 12 (even) connecting to all odd vertices
    # But that's 12 edges. Current solution uses 10 (only odd vertices 1,3,...,23)
    
    # Let me try a different approach: use a "path of hubs"
    # Or try to find a smaller set via greedy + local search
    
    # First, get a valid solution with the current approach (10 edges)
    added = []
    for v in range(1, n, 2):
        if (12, v) not in base_edges and (v, 12) not in base_edges:
            added.append((12, v))
    
    current_best = list(added)
    current_diam = compute_diameter(base_edges | set(current_best))
    
    # Try to improve: remove edges one by one and see if still valid
    improved = True
    while improved:
        improved = False
        for i in range(len(current_best)):
            candidate = current_best[:i] + current_best[i+1:]
            if len(candidate) == 0:
                continue
            diam = compute_diameter(base_edges | set(candidate))
            if diam <= 4:
                current_best = candidate
                improved = True
                break
    
    # Try adding different edges to potentially reduce further
    # Local search: try swapping edges
    best_solution = current_best
    best_count = len(current_best)
    
    # Try to find if we can do better with a different structure
    # Greedy: start empty, add edges that most reduce diameter
    # But this is complex. Let's try a few alternative constructions.
    
    # Alternative: two hubs
    # Hub A = 6 (even), Hub B = 17 (odd)
    # Connect A to nearby odds, B to nearby evens, and A-B
    alt1 = [(6, 17)]  # A-B
    for v in [5, 7, 9, 11, 13, 15]:  # odds near A
        if (6, v) not in base_edges:
            alt1.append((6, v))
    for v in [4, 8, 10, 12, 14, 16, 18, 20, 22]:  # evens near B
        if (v, 17) not in base_edges:
            alt1.append((v, 17))
    diam1 = compute_diameter(base_edges | set(alt1))
    if diam1 <= 4 and len(alt1) < best_count:
        best_solution = alt1
        best_count = len(alt1)
    
    # Alternative: three hubs
    # Hub1 = 4 (even), Hub2 = 12 (even), Hub3 = 20 (even)
    # Connect them in a path: 4-11, 12-11, 12-19, 20-19
    # Then connect each hub to nearby vertices
    alt2 = [(4, 11), (12, 11), (12, 19), (20, 19)]
    for v in [3, 5, 7, 9]:
        if (4, v) not in base_edges:
            alt2.append((4, v))
    for v in [11, 13, 15, 17]:
        if (12, v) not in base_edges:
            alt2.append((12, v))
    for v in [19, 21, 23]:
        if (20, v) not in base_edges:
            alt2.append((20, v))
    diam2 = compute_diameter(base_edges | set(alt2))
    if diam2 <= 4 and len(alt2) < best_count:
        best_solution = alt2
        best_count = len(alt2)
    
    # Try greedy construction from scratch
    greedy = []
    remaining_candidates = list(candidate_edges)
    for _ in range(20):  # Try up to 20 edges
        if time.time() > deadline:
            break
        best_edge = None
        best_diam = compute_diameter(base_edges | set(greedy))
        if best_diam <= 4:
            break
        for edge in remaining_candidates:
            test = greedy + [edge]
            d = compute_diameter(base_edges | set(test))
            if d < best_diam:
                best_diam = d
                best_edge = edge
            if d <= 4:
                best_edge = edge
                best_diam = d
                break
        if best_edge is None:
            break
        greedy.append(best_edge)
        remaining_candidates.remove(best_edge)
        if compute_diameter(base_edges | set(greedy)) <= 4 and len(greedy) < best_count:
            best_solution = greedy
            best_count = len(greedy)
    
    # Final: try to reduce best_solution
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_solution)):
            candidate = best_solution[:i] + best_solution[i+1:]
            if len(candidate) == 0:
                continue
            diam = compute_diameter(base_edges | set(candidate))
            if diam <= 4:
                best_solution = candidate
                improved = True
                break
    
    # Convert to required format
    result = [[int(u), int(v)] for u, v in best_solution]
    return result

# EVOLVE-BLOCK-END
