import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    trees = instance_trees(20, 611)
    n = 20

    # Process trees in reverse order (largest first)
    order = list(range(n - 1, -1, -1))

    used_edges = set()
    result = [None] * n

    # Precompute adjacency and degree for each tree
    tree_adj = [None] * n
    tree_deg = [None] * n
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for (a, b) in trees[i]:
            adj[a].append(b)
            adj[b].append(a)
            deg[a] += 1
            deg[b] += 1
        tree_adj[i] = adj
        tree_deg[i] = deg

    def embed_tree(tree_idx, used_edges_set, rng):
        """
        Try to embed tree tree_idx into K20 with edge-disjointness.
        Returns: list of k host vertices or None
        """
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for i in range(n):
                for j in range(i + 1, n):
                    e = (i, j)
                    if e not in used_edges_set:
                        return [i, j]
            return None

        adj = tree_adj[tree_idx]
        deg = tree_deg[tree_idx]

        # Pick root as highest degree vertex
        root = max(range(k), key=lambda v: deg[v])
        
        # BFS from root, tracking parent
        parent = [-1] * k
        visited = {root}
        queue = [root]
        assign_order = [root]
        while queue:
            u = queue.pop(0)
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    parent[w] = u
                    assign_order.append(w)
                    queue.append(w)
        
        # Precompute index mapping
        idx_of = [-1] * k
        for i, v in enumerate(assign_order):
            idx_of[v] = i
        
        host_map = [-1] * k
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                return True

            v = assign_order[pos]
            if pos == 0:
                candidates = [h for h in range(n) if h not in used_hosts]
            else:
                pv = parent[v]
                p_idx = idx_of[pv]
                ph = host_map[p_idx]
                candidates = [h for h in range(n) 
                              if h not in used_hosts 
                              and (min(h, ph), max(h, ph)) not in used_edges_set]

            for h in candidates:
                host_map[pos] = h
                used_hosts.add(h)
                if pos > 0:
                    pv = parent[v]
                    p_idx = idx_of[pv]
                    ph = host_map[p_idx]
                    edge = (min(h, ph), max(h, ph))
                    used_edges_set.add(edge)

                if backtrack(pos + 1):
                    return True

                if pos > 0:
                    pv = parent[v]
                    p_idx = idx_of[pv]
                    ph = host_map[p_idx]
                    edge = (min(h, ph), max(h, ph))
                    used_edges_set.discard(edge)
                used_hosts.discard(h)
                host_map[pos] = -1

            return False

        if backtrack(0):
            res = [0] * k
            for pos, v in enumerate(assign_order):
                res[v] = host_map[pos]
            return res
        return None

    # For each tree in order, try to embed
    rng = Random(42)
    for tree_idx in order:
        if time.time() > deadline:
            break

        k = tree_idx + 1
        res = embed_tree(tree_idx, used_edges, rng)

        if res is not None:
            result[tree_idx] = res
            for (a, b) in trees[tree_idx]:
                ha, hb = res[a], res[b]
                edge = (min(ha, hb), max(ha, hb))
                used_edges.add(edge)
        else:
            result[tree_idx] = None

    # Verify
    for i in range(n):
        if result[i] is not None:
            k = i + 1
            if len(result[i]) != k:
                result[i] = None
                continue
            if len(set(result[i])) != k:
                result[i] = None
                continue
            if any(v < 0 or v >= n for v in result[i]):
                result[i] = None
                continue

    return result
