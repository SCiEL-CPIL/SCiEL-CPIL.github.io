"""Packing trees T1..T20 into K20 via randomized backtracking."""
import time
import random
from collections import deque

_DEADLINE = 150.0
_MARGIN = 8.0  # safety margin before the 150s kill


def _instance_trees(n=20, seed=611):
    from random import Random
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


def _build_adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _order_vertices(adj, k, rnd=None):
    """Order vertices so that high-degree / central ones are placed first."""
    # degree
    deg = [len(adj[v]) for v in range(k)]
    # a simple centrality: sum of neighbor degrees
    nb_deg = [sum(deg[u] for u in adj[v]) for v in range(k)]
    # sort by (deg, nb_deg) descending; tie-break by index
    order = sorted(range(k), key=lambda v: (-deg[v], -nb_deg[v], v))
    return order


def _embed_tree(tree_edges, k, used_edges, order, rnd):
    """Try to embed a tree of k vertices into K20.
    Returns a list of 20 ints (image of vertex i) or None."""
    # randomize order slightly for diversity
    if rnd is not None and k > 2:
        # small perturbation: swap a couple of low-importance vertices
        # keep the structure mostly intact
        o = list(order)
        for _ in range(min(2, k // 3)):
            i = rnd.randrange(k)
            j = rnd.randrange(k)
            o[i], o[j] = o[j], o[i]
        order = o

    assign = [-1] * k
    # map tree vertex -> position in order (for reverse)
    # We place in order; assign[order[i]] = host
    host_used = [False] * 20

    # adjacency of tree
    adj = [[] for _ in range(k)]
    for a, b in tree_edges:
        adj[a].append(b)
        adj[b].append(a)

    # For a tree, when placing vertex v, its already-placed neighbors give
    # constraints on the host vertex.
    def backtrack(pos):
        if pos == k:
            return True
        v = order[pos]
        # determine candidate hosts: must be free, and if any neighbor of v is
        # already placed, the host must be adjacent (in K20 it's any different
        # vertex, so the only real constraint is that host != neighbor's host
        # is NOT required for edge-disjointness; edge-disjointness is about
        # edges, not vertices. So host can equal neighbor's host? No: an edge
        # (v, u) maps to (host(v), host(u)). If host(v)==host(u), that's a
        # loop, invalid. So host(v) must differ from host(u) for every placed
        # neighbor u.
        # Also the edge (host(v), host(u)) must not be in used_edges.
        candidates = []
        for h in range(20):
            if host_used[h]:
                continue
            ok = True
            for u in adj[v]:
                hu = assign[u]
                if hu != -1:
                    if h == hu:
                        ok = False
                        break
                    e = (min(h, hu), max(h, hu))
                    if e in used_edges:
                        ok = False
                        break
            if ok:
                candidates.append(h)
        if not candidates:
            return False
        # order candidates: prefer hosts with fewer already-used incident edges
        # (more "free" capacity) to leave room
        # compute for each h the number of used edges incident to h
        cand_order = sorted(candidates, key=lambda h: sum(1 for (a, b) in used_edges if a == h or b == h))
        # randomize tie-breaking a bit
        if rnd is not None:
            # shuffle within same count
            counts = {}
            for h in cand_order:
                c = sum(1 for (a, b) in used_edges if a == h or b == h)
                counts.setdefault(c, []).append(h)
            cand_order = []
            for c in sorted(counts.keys()):
                lst = counts[c]
                rnd.shuffle(lst)
                cand_order.extend(lst)
        for h in cand_order:
            assign[v] = h
            host_used[h] = True
            # add edges to used_edges temporarily
            added = []
            conflict = False
            for u in adj[v]:
                hu = assign[u]
                if hu != -1:
                    e = (min(h, hu), max(h, hu))
                    if e in used_edges:
                        conflict = True
                        break
                    used_edges.add(e)
                    added.append(e)
            if not conflict and backtrack(pos + 1):
                return True
            # undo
            for e in added:
                used_edges.discard(e)
            host_used[h] = False
            assign[v] = -1
        return False

    if backtrack(0):
        # assign is a permutation of some 20? No, k hosts chosen from 20.
        # Return list where position i = host of vertex i
        return assign
    return None


def _tree_sort_key(trees, idx):
    k = len(trees[idx])
    # larger k first (harder), and more edges first
    return (-k, -k + 1)


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    trees = _instance_trees(20, 611)
    # order of embedding: hardest first (largest k)
    order_idx = sorted(range(20), key=lambda i: -len(trees[i]))

    best = [None] * 20
    best_val = -1

    base_seed = 611
    restart = 0
    while time.time() < deadline:
        rnd = random.Random(base_seed + restart)
        used_edges = set()
        embeddings = [None] * 20
        ok = True
        for i in order_idx:
            if time.time() >= deadline:
                ok = False
                break
            k = len(trees[i])
            if k <= 1:
                embeddings[i] = [] if k == 1 else None
                # T1 is single vertex, no edges; represent as []? The contract:
                # entry k-1 is None or a list of k distinct ints. For k=1,
                # a list of 1 int. But seed returns None for k=1. We'll use
                # None for k=1 to match seed, or a single host. Let's use None
                # for k=1 (no edges to place) to be safe.
                embeddings[i] = None
                continue
            adj = _build_adjacency(trees[i], k)
            ordv = _order_vertices(adj, k, rnd)
            emb = _embed_tree(trees[i], k, used_edges, ordv, rnd)
            if emb is None:
                ok = False
                break
            embeddings[i] = emb
        if ok:
            # compute value
            val = 0
            for i in range(20):
                e = embeddings[i]
                if e is None:
                    continue
                # count edges of tree i
                val += len(trees[i])
            if val > best_val:
                best_val = val
                best = [list(e) if e is not None else None for e in embeddings]
            if best_val == 190:
                break
        restart += 1

    # fallback: seed behavior
    if best_val < 0:
        best = [None] * 19 + [list(range(20))]
        best_val = 19

    return best
