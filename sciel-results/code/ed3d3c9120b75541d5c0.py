import time
from random import Random
import heapq

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
    deadline = start_time + 140.0

    trees = instance_trees(20, 611)

    def norm_edges(ti):
        return [(min(a, b), max(a, b)) for a, b in trees[ti]]

    tree_order = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)

    def try_embed(ti, used_edges, rng, tlimit):
        k = ti + 1
        if k <= 1:
            return {}
        edges = norm_edges(ti)
        adj = [[] for _ in range(k)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # Root = highest degree
        root = max(range(k), key=lambda v: len(adj[v]))
        parent = {root: -1}
        children = [[] for _ in range(k)]
        visited = {root}
        queue = [root]
        for v in queue:
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = v
                    children[v].append(nb)
                    queue.append(nb)

        # Assignment order: priority by degree
        heap = [(-len(adj[root]), root)]
        assigned = set()
        assign_order = []
        while heap:
            neg_deg, v = heapq.heappop(heap)
            if v in assigned:
                continue
            assigned.add(v)
            assign_order.append(v)
            for c in children[v]:
                if c not in assigned:
                    heapq.heappush(heap, (-len(adj[c]), c))

        assignments = {}
        used_hosts = set()

        def bt(idx):
            if time.time() > tlimit:
                return False
            if idx == k:
                return True
            v = assign_order[idx]
            p = parent[v]
            candidates = [h for h in range(20) if h not in used_hosts]
            rng.shuffle(candidates)
            for h in candidates:
                if p != -1:
                    ph = assignments[p]
                    e = (min(h, ph), max(h, ph))
                    if e in used_edges:
                        continue
                assignments[v] = h
                used_hosts.add(h)
                if bt(idx + 1):
                    return True
                del assignments[v]
                used_hosts.discard(h)
            return False

        if bt(0):
            return assignments
        return None

    best_result = None
    best_edges = 0

    for seed in range(100):
        if time.time() > deadline - 15.0:
            break
        rng = Random(seed * 7919 + 12345)
        used_edges = set()
        embeddings = [None] * 20
        ok = True
        for ti in tree_order:
            if time.time() > deadline - 15.0:
                ok = False
                break
            k = ti + 1
            if k <= 1:
                embeddings[ti] = []
                continue
            res = try_embed(ti, used_edges, rng, deadline - 15.0)
            if res is None:
                ok = False
                break
            embeddings[ti] = [res[i] for i in range(k)]
            for a, b in norm_edges(ti):
                e = (min(res[a], res[b]), max(res[a], res[b]))
                used_edges.add(e)
        if ok:
            total = sum(len(trees[i]) for i in range(20))
            if total > best_edges:
                best_edges = total
                best_result = embeddings[:]
            break

    if best_result is None:
        best_result = [None] * 19 + [list(range(20))]

    final = []
    for i in range(20):
        if best_result[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in best_result[i]])
    return final
