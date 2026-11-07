import time
import random

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_constraint(edges, n):
    """Check if no simple 3-edge path has endpoints with equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u, v in edges:
        du = adj[u]
        dv = adj[v]
        for s in du:
            if s == v:
                continue
            ds = deg[s]
            for t in dv:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _edge_ok(adj, deg, u, v, n):
    """Check if adding edge (u,v) keeps constraint. Assumes (u,v) not present."""
    # new paths created by adding (u,v)
    # middle: s-u-v-t
    for s in adj[u]:
        if s == v:
            continue
        ds = deg[s]
        for t in adj[v]:
            if t == u or t == s:
                continue
            if ds == deg[t]:
                return False
    # first: u-v-x-y  (endpoints u, y)
    for x in adj[v]:
        if x == u:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[u] == deg[y]:
                return False
    # last: x-y-v-u (endpoints x, u)
    for x in adj[u]:
        if x == v:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[x] == deg[u]:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    n = 15
    rng = random.Random(12345)

    # Build a strong structured seed: near-threshold graph with distinct-ish degrees
    # Threshold graph construction: add vertices, each either isolated or dominating.
    # This yields a degree sequence with structure. We'll then local-search.
    # Strategy: start with a "complement of matching"-ish or use greedy distinct-degree target.

    # We'll use a greedy construction targeting a degree sequence with few collisions,
    # then local search (add, remove, swap) to maximize edges.

    best_edges = None
    best_count = -1

    def evaluate_and_track(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]

    # ---------- Initial construction via greedy add with distinct-degree bias ----------
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []

    # Try multiple random orderings, keep best
    best_seed_edges = []
    best_seed_count = -1

    for trial in range(40):
        if time.time() > deadline - 20:
            break
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if _edge_ok(adj, deg, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        if len(edges) > best_seed_count:
            best_seed_count = len(edges)
            best_seed_edges = [list(e) for e in edges]

    # Start local search from best seed
    edges = [tuple(e) for e in best_seed_edges]
    adj = _build_adj(edges, n)
    deg = [len(adj[i]) for i in range(n)]
    evaluate_and_track(edges)

    # ---------- Local search: add, remove, swap ----------
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    present = set()
    for u, v in edges:
        present.add((u, v))

    def add_edge(u, v):
        adj[u].add(v); adj[v].add(u)
        deg[u] += 1; deg[v] += 1
        edges.append((u, v))
        present.add((u, v))

    def remove_edge(u, v):
        adj[u].discard(v); adj[v].discard(u)
        deg[u] -= 1; deg[v] -= 1
        edges.remove((u, v))
        present.discard((u, v))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding missing edges
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if (u, v) in present:
                continue
            if _edge_ok(adj, deg, u, v, n):
                add_edge(u, v)
                evaluate_and_track(edges)
                improved = True
        # Try swaps: remove e1, add e2
        if time.time() < deadline:
            for (a, b) in list(edges):
                if time.time() > deadline:
                    break
                for (c, d) in all_pairs:
                    if (c, d) in present:
                        continue
                    if (a, b) == (c, d):
                        continue
                    # temp remove (a,b)
                    adj[a].discard(b); adj[b].discard(a)
                    deg[a] -= 1; deg[b] -= 1
                    edges.remove((a, b))
                    present.discard((a, b))
                    if _edge_ok(adj, deg, c, d, n):
                        add_edge(c, d)
                        evaluate_and_track(edges)
                        improved = True
                    else:
                        # restore
                        adj[a].add(b); adj[b].add(a)
                        deg[a] += 1; deg[b] += 1
                        edges.append((a, b))
                        present.add((a, b))

    result = []
    seen = set()
    for u, v in best_edges:
        u = int(u); v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    return result
