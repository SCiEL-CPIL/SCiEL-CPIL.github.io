import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def hopcroft_karp(n_left, n_right, adj):
        """Hopcroft-Karp maximum bipartite matching.
        adj: list of lists, adj[i] = list of right nodes connected to left node i.
        Returns matching as dict: left_node -> right_node.
        """
        INF = float('inf')
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == INF:
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        matching_size = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        return pair_u

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        m = len(edges)

        if m == 0:
            all_results.append([])
            continue

        # Build edge index
        edge_list = list(edges)
        edge_index = {e: i for i, e in enumerate(edge_list)}

        # Build adjacency for the bipartite matching problem
        # Left side: edge i can be "followed by" right side edge j
        # Edge i=(a,b) can precede edge j=(c,d) if:
        #   b==c or b==d (end of i matches start of j)
        #   a==c or a==d (start of i matches start of j) -- but we need directed traversal
        # Actually: edge i=(u,v) ends at v (or u). Edge j=(x,y) starts at x (or y).
        # We can chain i->j if one endpoint of i equals one endpoint of j.
        # More precisely: if we traverse i from u to v, then j must start at v.
        # But since edges are undirected, edge i can be traversed in either direction.
        # So edge i can precede edge j if they share a vertex.
        
        # However, for path decomposition, we need:
        # - Each edge appears in exactly one path
        # - Consecutive edges in a path share a vertex
        # - The path is a valid sequence of vertices
        
        # The standard reduction: create a bipartite graph where left nodes are edges,
        # right nodes are edges. Edge i -> j if i and j share a vertex.
        # Maximum matching gives maximum number of "links" between consecutive edges.
        # Min path count = m - max_matching_size.
        
        # But we need to be more careful: the matching must respect that
        # each edge can have at most one predecessor and one successor in the path.
        # This is exactly what a matching enforces.
        
        # However, we also need to ensure the resulting paths are valid (simple, using edges).
        # The matching approach gives us a set of linked edges; we then trace the paths.
        
        # Build adjacency: for each edge i, which edges j can follow i?
        # Edge i = (u, v). Edge j = (w, x). They can be linked if {u,v} ∩ {w,x} ≠ ∅.
        # But we need to be careful about the direction of traversal.
        
        # For simplicity, let's say edge i can precede edge j if they share at least one vertex.
        # This is a valid condition for chaining in a path decomposition.
        
        # Build left-to-right adjacency
        left_adj = [[] for _ in range(m)]
        for i in range(m):
            u1, v1 = edge_list[i]
            for j in range(m):
                if i == j:
                    continue
                u2, v2 = edge_list[j]
                # Check if edges share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    left_adj[i].append(j)

        # Run Hopcroft-Karp
        pair_u = hopcroft_karp(m, m, left_adj)
        
        # Reconstruct paths from the matching
        # pair_u[i] = j means edge i is followed by edge j in some path
        # We need to trace the chains
        
        # Build successor map: edge i -> edge j (if matched)
        succ = {}
        for i in range(m):
            if pair_u[i] != -1:
                succ[i] = pair_u[i]
        
        # Build predecessor map
        pred = {}
        for i in range(m):
            if i in succ:
                pred[succ[i]] = i
        
        # Find path starts (edges with no predecessor)
        starts = []
        for i in range(m):
            if i not in pred:
                starts.append(i)
        
        # Trace paths
        paths = []
        for start in starts:
            path_edges = []
            curr = start
            visited_in_path = set()
            while curr is not None and curr not in visited_in_path:
                visited_in_path.add(curr)
                path_edges.append(curr)
                curr = succ.get(curr, None)
            
            # Convert edge indices to vertex list
            if path_edges:
                vertex_path = []
                for ei in path_edges:
                    u, v = edge_list[ei]
                    if not vertex_path:
                        vertex_path = [u, v]
                    else:
                        last = vertex_path[-1]
                        if last == u:
                            vertex_path.append(v)
                        elif last == v:
                            vertex_path.append(u)
                        else:
                            # This shouldn't happen if matching is valid
                            # Fallback: just append
                            vertex_path.append(u)
                paths.append(vertex_path)
        
        # Verify all edges are covered
        covered_edges = set()
        for path in paths:
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                e = (min(u, v), max(u, v))
                covered_edges.add(e)
        
        if covered_edges != set(edges):
            # Fallback: use all edges as separate paths
            paths = [list(e) for e in edges]
        
        all_results.append(paths)

    return all_results
