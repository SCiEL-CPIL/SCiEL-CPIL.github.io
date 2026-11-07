"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from collections import defaultdict

    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # Cross adjacency: for each independent vertex v, which clique vertices it connects to
        cross_adj = {}
        for v in I:
            cross_adj[v] = adj[v] & set(C)

        # Total cross edges and inner edges
        total_cross = sum(len(cross_adj[v]) for v in I)
        total_inner = k * (k - 1) // 2

        # We want to maximize the number of triangles (v, u, w) where v in I, u,w in C
        # Each triangle uses 2 cross edges and 1 inner edge
        # Total cliques = total_cross + total_inner - 2 * num_triangles

        # Formulate as bipartite matching:
        # Left nodes: inner edges (u, w) with u < w, u,w in [0, k)
        # Right nodes: independent vertices v in I
        # Edge between (u,w) and v exists iff v is adjacent to both u and w
        # Maximum matching gives maximum number of triangles

        # Map inner edges to indices
        inner_edges = []
        inner_edge_idx = {}
        for u in range(k):
            for w in range(u + 1, k):
                idx = len(inner_edges)
                inner_edges.append((u, w))
                inner_edge_idx[(u, w)] = idx

        num_inner = len(inner_edges)
        num_indep = len(I)

        # Build bipartite graph: adjacency from left (inner edge idx) to right (indep vertex idx)
        # left_adj[i] = list of indep vertex indices that can form triangle with inner edge i
        left_adj = [[] for _ in range(num_inner)]
        # Also build right_adj for reverse lookup
        right_adj = [set() for _ in range(num_indep)]

        for v_idx, v in enumerate(I):
            S = cross_adj[v]  # set of clique vertices v connects to
            # For each pair (u, w) in S with u < w, add edge in bipartite graph
            S_list = sorted(S)
            for i in range(len(S_list)):
                u = S_list[i]
                for j in range(i + 1, len(S_list)):
                    w = S_list[j]
                    ie_idx = inner_edge_idx.get((u, w))
                    if ie_idx is not None:
                        left_adj[ie_idx].append(v_idx)
                        right_adj[v_idx].add(ie_idx)

        # Hopcroft-Karp for maximum cardinality matching
        # matchL[i] = right node matched to left node i, or -1
        # matchR[j] = left node matched to right node j, or -1
        matchL = [-1] * num_inner
        matchR = [-1] * num_indep
        dist = [0] * num_inner

        from collections import deque

        def bfs():
            q = deque()
            for u in range(num_inner):
                if matchL[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in left_adj[u]:
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return found

        def dfs(u):
            for v in left_adj[u]:
                mu = matchR[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False

        # Run Hopcroft-Karp
        matching_size = 0
        while bfs():
            for u in range(num_inner):
                if matchL[u] == -1:
                    if dfs(u):
                        matching_size += 1

        # Extract matched triangles
        triangles = []
        for u_idx in range(num_inner):
            if matchL[u_idx] != -1:
                u, w = inner_edges[u_idx]
                v = I[matchL[u_idx]]
                triangles.append((v, u, w))

        num_triangles = len(triangles)

        # Build clique partition
        cliques = []
        # Triangles
        cross_in_tri = set()
        inner_in_tri = set()
        for v, u, w in triangles:
            cliques.append([v, u, w])
            cross_in_tri.add((min(v, u), max(v, u)))
            cross_in_tri.add((min(v, w), max(v, w)))
            inner_in_tri.add((min(u, w), max(u, w)))

        # Remaining cross edges -> pairs
        for u, v in edges:
            if min(u, v) < k and max(u, v) >= k:
                key = (u, v)
                if key not in cross_in_tri:
                    cliques.append([u, v])

        # Remaining inner edges -> pairs
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in inner_in_tri:
                    cliques.append([u, v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE - 5:
            # Fallback: seed behavior (all edges as cliques)
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        cliques = solve_instance(*spec)
        results.append(cliques)

    # Ensure types are correct
    final_result = []
    for cliques in results:
        new_cliques = []
        for c in cliques:
            new_c = [int(x) for x in c]
            new_cliques.append(new_c)
        final_result.append(new_cliques)

    return final_result
# EVOLVE-BLOCK-END
