"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp(adj, n_left, n_right):
        """Hopcroft-Karp maximum bipartite matching.
        adj[u] = list of right nodes adjacent to left node u.
        Returns (matchL, matchR) where matchL[u] = right matched to u, matchR[v] = left matched to v.
        """
        INF = float('inf')
        matchL = [-1] * n_left
        matchR = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if matchL[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == INF:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found

        def dfs(u):
            for v in adj[u]:
                mu = matchR[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if matchL[u] == -1:
                    if dfs(u):
                        matching += 1
        return matchL, matchR, matching

    def path_decomposition(edges):
        """Find minimum path decomposition using bipartite matching."""
        if not edges:
            return []

        # Build adjacency structure
        edge_set = set()
        for e in edges:
            edge_set.add(tuple(e))
            edge_set.add((e[1], e[0]))  # both directions

        # Each undirected edge gives two directed stubs
        # Left side: all directed stubs (u,v)
        # Right side: all directed stubs (u,v)
        # They are the same set of stubs

        stubs = []
        stub_id = {}
        for e in edges:
            u, v = e
            stubs.append((u, v))
            stubs.append((v, u))
            stub_id[(u, v)] = len(stubs) - 1
            stub_id[(v, u)] = len(stubs) - 1

        n_stubs = len(stubs)

        # Build bipartite adjacency
        # Left stub (a,b) can connect to right stub (b,c) if (b,c) is a valid stub
        # i.e., {b,c} is an edge in the graph
        adj = [[] for _ in range(n_stubs)]
        for i, (u, v) in enumerate(stubs):
            # From stub (u,v) arriving at v, we can go to any stub (v,w) where {v,w} is an edge
            for j, (x, y) in enumerate(stubs):
                if x == v and j != i:
                    # stub (v, y) starts at v, so we can chain (u,v) -> (v,y)
                    adj[i].append(j)

        # Run Hopcroft-Karp
        matchL, matchR, matching_size = hopcroft_karp(adj, n_stubs, n_stubs)

        # Now reconstruct paths
        # Each matched pair (left i -> right j) means stub i is followed by stub j in a path
        # A path is a sequence of stubs where consecutive stubs are matched
        # Unmatched stubs are path endpoints

        # Build next pointer: next[i] = j if matchL[i] == j, else -1
        # prev[j] = i if matchR[j] == i, else -1
        nxt = [-1] * n_stubs
        prv = [-1] * n_stubs
        for i in range(n_stubs):
            if matchL[i] != -1:
                nxt[i] = matchL[i]
            if matchR[i] != -1:
                prv[i] = matchR[i]

        # Find all path starts (stubs with no predecessor)
        paths = []
        visited = set()
        for i in range(n_stubs):
            if prv[i] == -1 and i not in visited:
                # Start a path from stub i
                path_stubs = [i]
                cur = i
                while nxt[cur] != -1:
                    cur = nxt[cur]
                    path_stubs.append(cur)
                # Convert stubs to vertex path
                vertices = [stubs[path_stubs[0]][0]]
                for s in path_stubs:
                    vertices.append(stubs[s][1])
                paths.append(vertices)
                for s in path_stubs:
                    visited.add(s)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        paths = path_decomposition(edges)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END

