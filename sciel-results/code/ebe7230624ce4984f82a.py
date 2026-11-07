from random import Random
from math import comb
import time

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
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

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embedding_search(tree_edges, k, used_edges, host_n, rng, max_nodes):
    """Backtracking: assign each tree vertex 0..k-1 to distinct host vertices,
    ensuring no placed edge is in used_edges. Returns list of host vertices
    (index = tree vertex) or None."""
    if k == 1:
        # single vertex, no edges; any host vertex
        for h in range(host_n):
            return [h]
        return None
    if k == 2:
        # one edge (0,1)
        for a in range(host_n):
            for b in range(host_n):
                if a == b:
                    continue
                ek = _edge_key(a, b)
                if ek in used_edges:
                    continue
                return [a, b]
        return None

    adj = _build_adjacency(tree_edges, k)
    # order vertices by a DFS/BFS from 0 to give a nice parent structure
    order = [0]
    seen = {0}
    parent = {0: -1}
    stack = [0]
    while stack:
        v = stack.pop()
        for nb in adj[v]:
            if nb not in seen:
                seen.add(nb)
                parent[nb] = v
                order.append(nb)
                stack.append(nb)
    # order now is a spanning-tree order; each vertex (except root) has a parent in order
    # We'll assign in `order`; when assigning a vertex, its parent is already assigned.
    # Check edge (parent, v) not used.

    assign = [-1] * k
    assign[0] = None  # placeholder, set below
    placed = set()
    local_used = set()  # edges added by this embedding (subset of host edges)

    def backtrack(idx):
        if idx == len(order):
            return True
        v = order[idx]
        p = parent[v]
        if p == -1:
            # root
            cand = list(range(host_n))
        else:
            ph = assign[p]
            cand = [h for h in range(host_n) if h not in placed and _edge_key(ph, h) not in used_edges and _edge_key(ph, h) not in local_used]
        rng.shuffle(cand)
        for h in cand:
            assign[v] = h
            placed.add(h)
            if p != -1:
                local_used.add(_edge_key(ph, h))
            if backtrack(idx + 1):
                return True
            if p != -1:
                local_used.discard(_edge_key(ph, h))
            placed.discard(h)
            assign[v] = -1
        return False

    # root
    for h in range(host_n):
        assign[0] = h
        placed = {h}
        local_used = set()
        if backtrack(1):
            return assign
        assign[0] = -1
    return None

def construct():
    deadline = time.time() + 110.0  # safety margin under 120s
    host_n = 20
    total_edges = comb(host_n, 2)  # 190
    trees = instance_trees(20, 611)
    # indices of trees 0..19 correspond to T1..T20
    # We want to embed all 20; T1 has no edges.
    # Order: embed larger trees first (more constrained) -> better for backtracking success.
    order = sorted(range(20), key=lambda i: (len(trees[i]), i), reverse=True)

    best = [None] * 20
    best_count = -1

    for attempt_seed in range(100000):
        if time.time() > deadline:
            break
        rng = Random(attempt_seed)
        used_edges = set()
        result = [None] * 20
        ok = True
        for ti in order:
            k = ti + 1
            edges = trees[ti]
            if k == 1:
                # no edges; pick a free host vertex
                free = [h for h in range(host_n)]
                rng.shuffle(free)
                result[ti] = [free[0]]
                continue
            emb = _embedding_search(edges, k, used_edges, host_n, rng, max_nodes=10**7)
            if emb is None:
                ok = False
                break
            result[ti] = emb
            for (u, v) in edges:
                ek = _edge_key(emb[u], emb[v])
                if ek in used_edges:
                    ok = False
                    break
                used_edges.add(ek)
            if not ok:
                break
        if ok:
            # count embedded edges
            cnt = sum(len(trees[ti]) for ti in range(20))
            if cnt > best_count:
                best_count = cnt
                best = result
            if best_count == total_edges:
                # perfect
                break
    # Ensure types: lists of Python ints
    out = []
    for ti in range(20):
        e = best[ti]
        if e is None:
            out.append(None)
        else:
            out.append([int(x) for x in e])
    return out
