"""Structural edge-cover minimization via maximum matching on triangle-forming pairs."""
import time
import random
from collections import deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def trivial_solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        return [list(e) for e in edges]

    def hopcroft_karp(adj, num_left, num_right):
        """Maximum bipartite matching. adj[i] = list of right nodes adjacent to left node i."""
        match_l = [-1] * num_left
        match_r = [-1] * num_right
        dist = [0] * num_left

        def bfs():
            queue = deque()
            found = False
            for u in range(num_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if match_r[v] == -1:
                        found = True
                    elif dist[match_r[v]] == -1:
                        dist[match_r[v]] = dist[u] + 1
                        queue.append(match_r[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if match_r[v] == -1 or (dist[match_r[v]] == dist[u] + 1 and dfs(match_r[v])):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(num_left):
                if match_l[u] == -1 and dfs(u):
                    matching += 1
        return matching, match_l, match_r

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        m = len(edges)
        edge_set = set(edges)

        if time.time() > DEADLINE - 5.0:
            return trivial_solve(n, k, p, seed)

        # Identify all pairs of edges that form a triangle.
        # Build list of (edge_i, edge_j, triangle_vertices) for all valid triangle-forming pairs.
        # Then find maximum matching in the graph where nodes are edges and edges are triangle-forming pairs.
        
        # For efficiency, build adjacency: for each edge, which other edges form a triangle with it.
        # Triangle condition: two edges share exactly one vertex, and the two non-shared vertices are connected.
        
        # Build vertex-to-edges mapping
        by_vertex = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            by_vertex[u].append(idx)
            by_vertex[v].append(idx)
        
        # For each edge i, find all edges j that form a triangle with i.
        # i=(a,b), j=(c,d), share exactly one vertex s.
        # The other vertices are x (from i) and y (from j). Need (x,y) in edge_set.
        
        # To avoid duplicates, only consider j > i.
        # Build a bipartite graph: left side = edges, right side = edges, edge if they form triangle.
        # But this is not bipartite in general. However, we can use a different approach.
        
        # Alternative: model as maximum matching in a general graph.
        # Since m can be up to ~90*91/2 + 90*60*0.7 ≈ 4095 + 3780 ≈ 7875, general matching is expensive.
        
        # Better: exploit structure. In split graphs:
        # - Edges within clique: (u,v) with u,v < k
        # - Cross edges: (u,v) with u < k, v >= k
        # 
        # Triangle types:
        # 1. Two clique edges sharing a vertex: (u,v),(v,w) with u,v,w<k → always triangle (u,w exists)
        # 2. One clique edge (u,v) u,v<k and one cross edge (u,w) w>=k → triangle iff (v,w) exists
        # 3. Two cross edges: never form triangle (independent set)
        
        # So we can separate:
        # - Clique edges: C
        # - Cross edges: X
        # 
        # Triangles of type 1: pairs within C that share a vertex
        # Triangles of type 2: pairs (c,x) where c=(u,v) in C, x=(u,w) in X, and (v,w) in edge_set
        
        # Maximum matching on the full graph is complex. Let's use a simpler approach:
        # Greedy with backtracking or use a heuristic that is better than pure greedy.
        
        # Actually, let's try: build the triangle-forming graph and use a simple maximum matching algorithm.
        # For m up to ~8000, we can try a randomized greedy with multiple restarts.
        
        # Build adjacency: tri_adj[i] = list of j > i such that edges[i] and edges[j] form a triangle.
        tri_adj = [[] for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for s in (u1, v1):
                for j in by_vertex[s]:
                    if j <= i:
                        continue
                    u2, v2 = edges[j]
                    # Check they share exactly one vertex
                    shared = set((u1, v1)) & set((u2, v2))
                    if len(shared) != 1:
                        continue
                    a = u1 if u1 != s else v1
                    b = u2 if u2 != s else v2
                    if a == b:
                        continue
                    if (min(a, b), max(a, b)) in edge_set:
                        tri_adj[i].append(j)
        
        # Now find maximum matching in this graph (which is not necessarily bipartite).
        # Use a simple greedy with randomization and multiple restarts.
        
        best_pairs = []
        best_count = 0
        
        num_restarts = 20
        time_per_restart = (DEADLINE - time.time()) / num_restarts - 0.1
        if time_per_restart < 0.1:
            num_restarts = 1
            time_per_restart = max(0.1, DEADLINE - time.time() - 0.5)
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 2.0:
                break
            
            matched = [False] * m
            pairs = []
            order = list(range(m))
            random.Random(seed + restart * 1000).shuffle(order)
            
            for i in order:
                if matched[i]:
                    continue
                for j in tri_adj[i]:
                    if matched[j]:
                        continue
                    matched[i] = True
                    matched[j] = True
                    u1, v1 = edges[i]
                    u2, v2 = edges[j]
                    shared = set((u1, v1)) & set((u2, v2))
                    s = next(iter(shared))
                    a = u1 if u1 != s else v1
                    b = u2 if u2 != s else v2
                    pairs.append((i, j, (s, a, b)))
                    break
            
            count = len(pairs)
            if count > best_count:
                best_count = count
                best_pairs = pairs[:]
        
        # Build partition
        matched = [False] * m
        for (i, j, tri) in best_pairs:
            matched[i] = True
            matched[j] = True
        
        cliques = []
        for (i, j, tri) in best_pairs:
            cliques.append(list(tri))
        for idx in range(m):
            if not matched[idx]:
                cliques.append(list(edges[idx]))
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        results.append(solve(*spec))

    out = []
    for lst in results:
        cl = []
        for c in lst:
            cl.append([int(x) for x in c])
        out.append(cl)
    return out

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
