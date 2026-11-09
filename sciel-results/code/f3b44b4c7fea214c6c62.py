"""Packing trees T1..T20 into K20 using backtracking edge-disjoint embedding."""
import time
from random import Random


def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _build_tree_info(tree_edges):
    """Build adjacency and degree info for a tree."""
    if not tree_edges:
        return {}, {}, []
    verts = set()
    for u, v in tree_edges:
        verts.add(u)
        verts.add(v)
    verts = sorted(verts)
    adj = {v: [] for v in verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    degree = {v: len(adj[v]) for v in verts}
    return adj, degree, verts


def _embed_with_backtracking(tree_edges, n, used_edges, rng, max_nodes=100000):
    """
    Embed tree_edges into K_n using backtracking with LCV heuristic.
    """
    if not tree_edges:
        return {}

    adj, degree, verts = _build_tree_info(tree_edges)
    k = len(verts)

    leaves = [v for v in verts if degree[v] == 1]
    if leaves:
        root = rng.choice(leaves)
    else:
        root = verts[0]

    order = [root]
    parent = {root: -1}
    visited = {root}
    queue = [root]
    idx = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                parent[w] = u
                queue.append(w)
                order.append(w)

    # Precompute available neighbors for each host vertex
    # neighbor_list[h] contains all x such that (h,x) is NOT in used_edges
    neighbor_list = [[] for _ in range(n)]
    for h in range(n):
        for x in range(n):
            if h == x: continue
            e = (min(h, x), max(h, x))
            if e not in used_edges:
                neighbor_list[h].append(x)

    mapping = {}
    host_used = set()
    node_count = [0]

    def backtrack(pos):
        if node_count[0] > max_nodes:
            return False
        node_count[0] += 1

        if pos == len(order):
            return True

        v = order[pos]
        p = parent[v]

        if p == -1:
            candidates = [h for h in range(n) if h not in host_used]
        else:
            hp = mapping[p]
            candidates = [h for h in range(n)
                          if h not in host_used
                          and (min(hp, h), max(hp, h)) not in used_edges]

        if not candidates:
            return False

        # LCV Heuristic: Sort by number of available neighbors for future placements
        # Available neighbors: x in neighbor_list[h] such that x not in host_used
        def heuristic_score(h):
            count = 0
            for x in neighbor_list[h]:
                if x not in host_used:
                    count += 1
            return -count # Minimize count? No, maximize count. So sort descending.
            
        # To add randomness, we can shuffle the candidates before sorting?
        # Or just sort. The random root choice provides diversity.
        candidates.sort(key=heuristic_score, reverse=True)

        for h in candidates:
            mapping[v] = h
            host_used.add(h)
            if backtrack(pos + 1):
                return True
            del mapping[v]
            host_used.discard(h)

        return False

    for attempt in range(10):
        mapping.clear()
        host_used.clear()
        node_count[0] = 0
        if backtrack(0):
            return dict(mapping)

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()
    result = [None] * 20

    # Process trees in order of decreasing number of edges (most constrained first)
    # But we need to return in original order
    order = list(range(20))
    order.sort(key=lambda i: len(trees[i]), reverse=True)

    # Try multiple global random seeds
    best_total = 0
    best_result = [None] * 20

    for global_seed in range(50):
        if time.time() > deadline - 3.0:
            break

        rng = Random(global_seed)
        used_edges = set()
        result = [None] * 20
        success_count = 0

        for idx in order:
            if time.time() > deadline - 2.0:
                break

            k = idx + 1
            tree_edges = trees[idx]

            if not tree_edges:
                # Tree with 1 vertex (k=1)
                result[idx] = [0]
                success_count += 1
                continue

            mapping = _embed_with_backtracking(tree_edges, n, used_edges, rng)
            if mapping is not None:
                result_list = []
                for v in range(k):
                    result_list.append(int(mapping[v]))
                result[idx] = result_list
                success_count += 1

                for u, v in tree_edges:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    used_edges.add(e)

        # Calculate total edges embedded
        total_edges = sum(len(trees[i]) for i in range(20) if result[i] is not None)

        if total_edges > best_total:
            best_total = total_edges
            best_result = [list(r) if r is not None else None for r in result]

    return best_result
