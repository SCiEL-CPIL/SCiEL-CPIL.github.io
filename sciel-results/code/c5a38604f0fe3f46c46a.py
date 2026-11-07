import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # All possible edges (not in base)
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in base_edges:
                all_edges.append((i, j))
    
    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for e in base_edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        for e in edge_list:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        return adj
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        return True
    
    def diameter(adj):
        # BFS from each vertex
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if -1 in dist:
                return -1  # disconnected
            max_d = max(max_d, max(dist))
        return max_d
    
    def valid(edge_list):
        adj = build_adj(edge_list)
        if not is_triangle_free(adj):
            return False
        d = diameter(adj)
        return d != -1 and d <= 4
    
    # Greedy approach: try to find minimum edges
    # First, try small numbers of edges
    
    # Pre-compute: for each edge, check if adding it to base keeps triangle-free
    # Use a smarter search
    
    # Try 6, 7, 8, 9 edges in order
    # For efficiency, use a candidate set
    
    # Key insight: edges that are "long" are more useful for reducing diameter
    # Sort candidates by usefulness (distance in path)
    
    candidate_edges = sorted(all_edges, key=lambda e: e[1] - e[0], reverse=True)
    
    # Limit candidates to those with path distance >= 3 (distance 2 creates triangles)
    # Actually distance 2: (i, i+2) - path i-(i+1)-(i+2), adding (i,i+2) creates triangle
    # So we must exclude edges with path distance 2
    safe_candidates = [e for e in candidate_edges if (e[1] - e[0]) != 2]
    
    # Also, we need to check triangle-freedom with combinations
    
    # Try greedy: iteratively add the edge that most reduces diameter
    # But first, let's try a few fixed constructions
    
    # Construction 1: 8-edge solution attempt
    # Use two "rings" or a grid-like structure
    
    # Let me try: connect vertices to create a structure with diameter 4
    # Vertices 0-23. If I can make it so every vertex is within 2 of a "center" 
    # and centers are within 2 of each other...
    
    # Try: hubs at 6 and 17
    # 6 connects to: 0,2,4,8,10,12 (even, dist>=2 from 6, not 5 or 7)
    # Wait, 6 is even. Connecting to even vertices: dist in path is even.
    # (6,0): dist 6, no triangle issue directly
    # But need to check all triangles.
    
    # Let me just do a bounded search for small k
    
    best = None
    
    # Try k from 6 upward
    for k in range(6, 12):
        if time.time() > deadline:
            break
        
        # For small k, try combinations from top candidates
        # Limit to top ~20 candidates for combinatorial search
        top_cands = safe_candidates[:min(25, len(safe_candidates))]
        
        found = False
        for combo in combinations(top_cands, k):
            if time.time() > deadline:
                break
            if valid(list(combo)):
                best = list(combo)
                found = True
                break
        if found:
            break
    
    if best is None:
        # Fallback: use the known 10-edge solution
        best = [(12, v) for v in [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]]
    
    # Convert to list of lists
    result = [[int(u), int(v)] for u, v in best]
    return result
