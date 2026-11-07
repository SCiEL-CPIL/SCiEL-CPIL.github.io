import time
import random
from random import Random
from collections import deque

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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    trees = instance_trees(20, 611)
    N = 20

    # Pre-calculate tree structures
    tree_structures = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        if k == 1:
            tree_structures.append({
                'k': k,
                'edges': [],
                'adj': [[]],
                'deg': [0],
                'var_order': [0],
                'bfs_order': [0]
            })
            continue
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)

        # Sort vertices by degree descending for MRV
        var_order = sorted(range(k), key=lambda v: -deg[v])

        # BFS order from a leaf
        start_v = 0
        for v in range(k):
            if deg[v] == 1:
                start_v = v
                break
        bfs_order = [start_v]
        visited = {start_v}
        queue = deque([start_v])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    bfs_order.append(v)
                    queue.append(v)

        tree_structures.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'var_order': var_order,
            'bfs_order': bfs_order
        })

    def calculate_host_remaining_deg(used_edges):
        host_deg = [0] * N
        for u, v in used_edges:
            host_deg[u] += 1
            host_deg[v] += 1
        return [N - 1 - d for d in host_deg]

    def embed_tree(tree_idx, used_edges, rng, node_limit=300000, use_bfs=False):
        struct = tree_structures[tree_idx]
        k = struct['k']
        adj = struct['adj']
        var_order = struct['var_order'] if not use_bfs else struct['bfs_order']

        if k == 1:
            return [0]

        mapping = [-1] * k
        used_hosts = set()
        nodes_explored = [0]
        host_rem_deg = calculate_host_remaining_deg(used_edges)

        # Pre-compute neighbor indices for each position in var_order
        neighbor_indices = [[] for _ in range(k)]
        for i in range(k):
            v = var_order[i]
            for j in range(i):
                if var_order[j] in adj[v]:
                    neighbor_indices[i].append(j)

        def backtrack(idx):
            if nodes_explored[0] > node_limit:
                return False
            if idx == k:
                return True

            v = var_order[idx]
            cands = []
            for h in range(N):
                if h in used_hosts:
                    continue
                ok = True
                for j in neighbor_indices[idx]:
                    w = var_order[j]
                    mw = mapping[w]
                    if mw != -1:
                        edge = (min(h, mw), max(h, mw))
                        if edge in used_edges:
                            ok = False
                            break
                if ok:
                    cands.append(h)

            if not cands:
                return False

            # Heuristic: sort by remaining degree descending, with random tie-breaking
            cands.sort(key=lambda h: (-host_rem_deg[h], rng.random()))

            for h in cands:
                nodes_explored[0] += 1
                if nodes_explored[0] > node_limit:
                    return False

                mapping[v] = h
                used_hosts.add(h)

                added_edges = []
                for j in neighbor_indices[idx]:
                    w = var_order[j]
                    mw = mapping[w]
                    e = (min(h, mw), max(h, mw))
                    used_edges.add(e)
                    added_edges.append(e)

                if backtrack(idx + 1):
                    return True

                for e in added_edges:
                    used_edges.discard(e)
                used_hosts.discard(h)
                mapping[v] = -1

            return False

        if backtrack(0):
            return [mapping[i] for i in range(k)]
        return None

    best_result = None
    best_edges = -1

    # Process trees in order of decreasing size (most constrained first)
    order = list(range(19, -1, -1))

    # Multiple strategies with different random seeds
    for attempt in range(20):
        if time.time() > deadline - 5:
            break

        rng = Random(611 + attempt)
        used_edges = set()
        embedding = [None] * N
        total_edges = 0
        success = True

        # Alternate between MRV and BFS ordering for variety
        use_bfs = (attempt % 3 == 0)

        for idx in order:
            if time.time() > deadline - 5:
                success = False
                break

            k = idx + 1
            if k == 1:
                embedding[idx] = [0]
                continue

            m = embed_tree(idx, used_edges, rng, node_limit=300000, use_bfs=use_bfs)
            if m is None:
                success = False
                break

            embedding[idx] = m
            tree_edges = trees[idx]
            for u, v in tree_edges:
                host_u = m[u]
                host_v = m[v]
                e = (min(host_u, host_v), max(host_u, host_v))
                used_edges.add(e)
            total_edges += len(tree_edges)

        if success and total_edges > best_edges:
            best_edges = total_edges
            best_result = embedding
            if best_edges == 190:
                break

    if best_result is not None:
        return [list(e) if e is not None else None for e in best_result]

    # Fallback
    return [None] * 19 + [list(range(20))]

if __name__ == "__main__":
    result = construct()
    print(result)
