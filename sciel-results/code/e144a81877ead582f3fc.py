"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
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
    even_vertices = [v for v in range(n) if v % 2 == 0]
    odd_vertices = [v for v in range(n) if v % 2 == 1]
    
    # Candidate edges: all (even, odd) pairs not already in base_edges
    candidate_edges = []
    for u in even_vertices:
        for v in odd_vertices:
            if (u, v) not in base_edges:
                candidate_edges.append((u, v))
    
    def compute_diameter(edge_set):
        adj_local = [set() for _ in range(n)]
        for u, v in edge_set:
            adj_local[u].add(v)
            adj_local[v].add(u)
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
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    def is_valid(edge_list):
        if len(edge_list) == 0:
            return compute_diameter(base_edges) <= 4
        return compute_diameter(base_edges | set(edge_list)) <= 4
    
    # Strategy: Try to find minimal edge set using systematic search
    # The path P_24 has diameter 23. We need to reduce it to <= 4.
    
    # Key insight: With k added edges, the maximum possible diameter reduction is limited.
    # A hub connected to all vertices on one side of the bipartition gives diameter 4.
    # Vertex 12 (even) connected to all odd vertices: 12 edges, but path edges already connect some.
    
    # Let's think about what's actually needed:
    # If we add edge (12, v) for odd v, then:
    # - dist(0, 23) = min(path_dist, dist(0,12)+1+dist(23,v)+1) for some v
    # Actually, with hub 12 connected to all odds:
    #   dist(0, x) <= min(x, 12 + 1 + |x-12| for odd x, or 12+1+|x-12|+1 for even x)
    #   Wait, from 0 to even vertex 2k: path dist = 2k. Via hub: 0->12 (12 steps) -> odd neighbor (1) -> ... 
    #   Actually 0 is even, 12 is even. dist(0,12)=12 on path.
    #   If 12 connects to all odds, then dist(0, 23) = min(23, 12+1+1) = 14? No.
    #   0->1...12 is 12 steps. 12->23 (odd) is 1 step. So dist(0,23) = 13.
    #   That's still > 4.
    
    # So a single hub isn't enough. We need multiple hubs or a different structure.
    
    # Let me think more carefully. Diameter <= 4 means every pair of vertices is at distance <= 4.
    # On a path of 24 vertices, the endpoints 0 and 23 are at distance 23.
    # To reduce dist(0,23) to <= 4, we need a shortcut of length at most 4.
    # This means there must be a path from 0 to 23 using at most 4 edges in the augmented graph.
    # Since 0 and 23 are at opposite ends, we need "jump" edges.
    
    # For diameter <= 4, we need:
    # - Every vertex within distance 2 of some "center" structure
    # - Or more precisely, the graph has radius <= 2 (if there's a center) or diameter 4 with no single center
    
    # Let me try a different approach: use a small number of "long" edges to create shortcuts.
    # For example:
    # Edge (0, 20): dist(0,23) <= 1 + 3 = 4 (via 20->21->22->23)
    # Edge (2, 22): dist(0,23) <= 2 + 1 + 1 = 4 (0->1->2, 2->22, 22->23)
    # But we need ALL pairs within distance 4, not just endpoints.
    
    # Let me try to construct a solution with a small number of edges and verify.
    
    # Approach: Use edges that create a "diameter-reducing" structure.
    # Try: connect pairs that are far apart on the path.
    
    # Let's try a systematic greedy approach:
    # Start with the base path. Find the pair with maximum distance.
    # Add an edge that reduces the maximum distance the most.
    
    best_solution = None
    best_count = float('inf')
    
    # Greedy: iteratively add the edge that most reduces the diameter
    current_edges = []
    current_diam = n - 1  # 23
    
    # Precompute all candidate edges
    all_candidates = list(candidate_edges)
    
    for iteration in range(15):  # Try up to 15 edges
        if time.time() > deadline:
            break
        if current_diam <= 4:
            break
        
        best_edge = None
        best_new_diam = current_diam
        
        for edge in all_candidates:
            if edge in current_edges:
                continue
            test_edges = current_edges + [edge]
            d = compute_diameter(base_edges | set(test_edges))
            if d < best_new_diam:
                best_new_diam = d
                best_edge = edge
            if d <= 4:
                best_edge = edge
                best_new_diam = d
                break
        
        if best_edge is None:
            break
        
        current_edges.append(best_edge)
        current_diam = best_new_diam
        
        if current_diam <= 4 and len(current_edges) < best_count:
            best_solution = list(current_edges)
            best_count = len(current_edges)
    
    # Now try to reduce the solution by removing edges
    if best_solution is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                candidate = best_solution[:i] + best_solution[i+1:]
                if is_valid(candidate):
                    best_solution = candidate
                    best_count = len(candidate)
                    improved = True
                    break
    
    # Also try some specific constructions
    
    # Construction 1: Two "rings" or "stars"
    # Connect vertex 6 to vertices 0,4,8,12,16,20 (all even, distance 6 apart)
    # Connect vertex 17 to vertices 1,5,9,13,17,21 (all odd)
    # Plus edge (6,17)
    alt1 = [(6, 17)]
    for v in [0, 4, 8, 12, 16, 20]:
        if v != 6 and (6, v) not in base_edges:
            alt1.append((6, v))
    for v in [1, 5, 9, 13, 21]:
        if v != 17 and (17, v) not in base_edges:
            alt1.append((17, v))
    
    d1 = compute_diameter(base_edges | set(alt1))
    if d1 <= 4 and len(alt1) < best_count:
        best_solution = alt1
        best_count = len(alt1)
        # Try to reduce
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                candidate = best_solution[:i] + best_solution[i+1:]
                if is_valid(candidate):
                    best_solution = candidate
                    best_count = len(candidate)
                    improved = True
                    break
    
    # Construction 2: Three hubs
    # Hub A = 4, Hub B = 12, Hub C = 20
    # Connect A-B via odd vertex 8: (4,8), (8,12)
    # Connect B-C via odd vertex 16: (12,16), (16,20)
    # Connect each hub to nearby vertices
    alt2 = [(4, 8), (8, 12), (12, 16), (16, 20)]
    # A=4 covers: 0,2,4,6,8 (dist <= 2 from 4 or via edges)
    # Need to ensure all vertices within dist 4 of each other
    
    d2 = compute_diameter(base_edges | set(alt2))
    if d2 <= 4 and len(alt2) < best_count:
        best_solution = alt2
        best_count = len(alt2)
    
    # Construction 3: Direct long edges
    # (0, 20), (2, 22), (4, 18), (6, 16), (8, 14), (10, 12)
    # Check if this gives diameter <= 4
    alt3 = [(0, 20), (2, 22), (4, 18), (6, 16), (8, 14), (10, 12)]
    d3 = compute_diameter(base_edges | set(alt3))
    if d3 <= 4 and len(alt3) < best_count:
        best_solution = alt3
        best_count = len(alt3)
    
    # Construction 4: Fewer, more strategic edges
    # (0, 18), (2, 22), (6, 14), (8, 20), (4, 16), (10, 12)
    alt4 = [(0, 18), (2, 22), (6, 14), (8, 20), (4, 16), (10, 12)]
    d4 = compute_diameter(base_edges | set(alt4))
    if d4 <= 4 and len(alt4) < best_count:
        best_solution = alt4
        best_count = len(alt4)
    
    # Construction 5: Try (0,16), (2,20), (4,22), (6,18), (8,14), (10,12)
    alt5 = [(0, 16), (2, 20), (4, 22), (6, 18), (8, 14), (10, 12)]
    d5 = compute_diameter(base_edges | set(alt5))
    if d5 <= 4 and len(alt5) < best_count:
        best_solution = alt5
        best_count = len(alt5)
    
    # Construction 6: Even fewer edges
    # (0, 20), (2, 22), (4, 18), (8, 14)
    alt6 = [(0, 20), (2, 22), (4, 18), (8, 14)]
    d6 = compute_diameter(base_edges | set(alt6))
    if d6 <= 4 and len(alt6) < best_count:
        best_solution = alt6
        best_count = len(alt6)
    
    # Construction 7: (0, 18), (2, 22), (6, 16), (8, 20)
    alt7 = [(0, 18), (2, 22), (6, 16), (8, 20)]
    d7 = compute_diameter(base_edges | set(alt7))
    if d7 <= 4 and len(alt7) < best_count:
        best_solution = alt7
        best_count = len(alt7)
    
    # Construction 8: (0, 16), (4, 22), (6, 20), (10, 14)
    alt8 = [(0, 16), (4, 22), (6, 20), (10, 14)]
    d8 = compute_diameter(base_edges | set(alt8))
    if d8 <= 4 and len(alt8) < best_count:
        best_solution = alt8
        best_count = len(alt8)
    
    # Construction 9: (0, 14), (2, 20), (6, 18), (8, 22), (4, 16)
    alt9 = [(0, 14), (2, 20), (6, 18), (8, 22), (4, 16)]
    d9 = compute_diameter(base_edges | set(alt9))
    if d9 <= 4 and len(alt9) < best_count:
        best_solution = alt9
        best_count = len(alt9)
    
    # Construction 10: Try to be even more minimal
    # (0, 12), (4, 20), (8, 22), (2, 18), (6, 14), (10, 16)
    alt10 = [(0, 12), (4, 20), (8, 22), (2, 18), (6, 14), (10, 16)]
    d10 = compute_diameter(base_edges | set(alt10))
    if d10 <= 4 and len(alt10) < best_count:
        best_solution = alt10
        best_count = len(alt10)
    
    # If we still haven't found a good solution, use the greedy result
    if best_solution is None:
        # Fallback: use hub construction
        best_solution = []
        for v in range(1, n, 2):
            if (12, v) not in base_edges:
                best_solution.append((12, v))
        best_count = len(best_solution)
    
    # Final reduction pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_solution)):
            candidate = best_solution[:i] + best_solution[i+1:]
            if is_valid(candidate):
                best_solution = candidate
                best_count = len(candidate)
                improved = True
                break
    
    # Convert to required format
    result = [[int(u), int(v)] for u, v in best_solution]
    return result

# EVOLVE-BLOCK-END
