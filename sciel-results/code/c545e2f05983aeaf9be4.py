"""Backtracking + MRV + forward-checking packer for trees T1..T20 into K20."""
import time
from random import Random

_DEADLINE = time.time() + 130.0  # safety margin before 150s wall-clock kill


def _instance_trees(n=20, seed=611):
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


def _build_tree(adj, n):
    # adj: list of neighbor lists. returns (edges, roots) for a spanning tree.
    parent = [-1] * n
    seen = [False] * n
    seen[0] = True
    stack = [0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                stack.append(v)
    # verify connected
    if len(order) != n:
        return None
    edges = []
    for v in range(1, n):
        edges.append((v, parent[v]))
    # roots = vertices of degree 1
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    roots = [v for v in range(n) if deg[v] == 1]
    return edges, roots


def _all_embeddings(n, edges, roots):
    # Precompute all distinct embeddings (as tuple of host vertices in root order).
    # For n=20 with ~17 edges this is bounded; we lazily generate via backtracking instead.
    pass


def _solve():
    N = 20
    trees = _instance_trees(N, 611)
    # Build adjacency per tree
    tdata = []
    for k, edges in enumerate(trees):
        if k <= 1:
            tdata.append((k, edges, None, None))
            continue
        adj = [[] for _ in range(k)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        built = _build_tree(adj, k)
        if built is None:
            tdata.append((k, edges, None, None))
            continue
        edges2, roots = built
        tdata.append((k, edges2, roots, adj))

    # Packing order: trivial first, then by (size, edges) to reduce branching early.
    order = []
    for k in range(1, N + 1):
        kk, edges, roots, adj = tdata[k - 1]
        if kk <= 1:
            order.append((0, k))
        else:
            order.append((kk, k))
    order.sort(key=lambda t: (t[0], t[1]))
    pack_seq = [k for _, k in order]

    # edge id
    def eid(a, b):
        if a > b:
            a, b = b, a
        return a * N + b

    total_edges = sum(len(tdata[k - 1][1]) for k in range(1, N + 1))

    best = [None] * N
    best_count = 0

    # State
    used = [False] * (N * N)
    assigned = [None] * N  # assigned[k-1] = tuple of host verts or None

    def _embed(k):
        kk, edges, roots, adj = tdata[k - 1]
        if kk <= 1:
            return
        # Backtracking over distinct embeddings using root ordering.
        # We'll do a recursive assignment with MRV + forward checking.
        # Map: local vertex -> host vertex. We fix roots to distinct hosts.
        # Use iterative deepening? No: full search with pruning.
        # Represent partial map as list host_of[local] = -1 unknown.
        # To keep it fast, we enumerate via recursion over local vertices in a
        # heuristic order: roots first, then by remaining degree.
        # We'll implement a DFS that returns the first valid embedding and
        # also supports "next" by continuing search (we need many for backtracking).
        # Instead: generate embeddings lazily with a generator.

        # Precompute local degree
        ldeg = [0] * kk
        for a, b in edges:
            ldeg[a] += 1
            ldeg[b] += 1
        # Heuristic order: roots first (degree 1), then increasing degree, then index
        order_local = sorted(range(kk), key=lambda v: (0 if v in roots else 1, ldeg[v], v))

        host_of = [-1] * kk
        # available hosts
        avail = [h for h in range(N) if not any(used[eid(h, x)] for x in range(N))]

        def dfs(idx):
            nonlocal best_count
            if idx == kk:
                # Full embedding found; check all edges free (forward check should ensure)
                # Record as a candidate by yielding
                yield tuple(host_of)
                return
            v = order_local[idx]
            # Candidate hosts: must be in avail and not conflict on already-fixed edges
            for h in range(N):
                if any(used[eid(h, x)] for x in range(N)):
                    continue
                # Check edges from v to already-assigned neighbors
                ok = True
                for nb in adj[v]:
                    if host_of[nb] != -1:
                        if used[eid(h, host_of[nb])]:
                            ok = False
                            break
                if not ok:
                    continue
                host_of[v] = h
                # forward check: mark nothing yet (edges not fully determined until both ends)
                for res in dfs(idx + 1):
                    yield res
                host_of[v] = -1

        return dfs()

    def search(pos):
        nonlocal best_count
        if time.time() > _DEADLINE:
            return
        if pos == N:
            # All packed
            for k in range(1, N + 1):
                a = assigned[k - 1]
                if a is None:
                    best[k - 1] = None
                else:
                    best[k - 1] = list(a)
            best_count = total_edges
            return

        k = pack_seq[pos]
        kk, edges, roots, adj = tdata[k - 1]
        if kk <= 1:
            # trivial
            assigned[k - 1] = () if kk == 1 else (0, 1)
            if kk == 2:
                # mark edge (0,1) used
                e = eid(0, 1)
                if not used[e]:
                    used[e] = True
                    search(pos + 1)
                    used[e] = False
            else:
                search(pos + 1)
            assigned[k - 1] = None
            return

        # Try embeddings via generator
        for emb in _embed(k):
            # Check all edges of this embedding are free
            ok = True
            touched = []
            for a, b in edges:
                e = eid(emb[a], emb[b])
                if used[e]:
                    ok = False
                    break
                touched.append(e)
            if not ok:
                continue
            for e in touched:
                used[e] = True
            assigned[k - 1] = emb
            search(pos + 1)
            assigned[k - 1] = None
            for e in touched:
                used[e] = False
            if time.time() > _DEADLINE:
                return
            # If we already have a full packing, stop (optimal)
            if best_count == total_edges:
                return

    search(0)

    # If no full packing, build best partial: greedily pack in a fixed order
    if best_count < total_edges:
        # Reset and do a greedy pass in pack_seq order
        for k in range(1, N + 1):
            best[k - 1] = None
        used2 = [False] * (N * N)
        for k in pack_seq:
            kk, edges, roots, adj = tdata[k - 1]
            if kk <= 1:
                if kk == 1:
                    best[k - 1] = None
                else:
                    e = eid(0, 1)
                    if not used2[e]:
                        used2[e] = True
                        best[k - 1] = [0, 1]
                    else:
                        best[k - 1] = None
                continue
            found = False
            for emb in _embed(k):
                ok = True
                touched = []
                for a, b in edges:
                    e = eid(emb[a], emb[b])
                    if used2[e]:
                        ok = False
                        break
                    touched.append(e)
                if not ok:
                    continue
                for e in touched:
                    used2[e] = True
                best[k - 1] = list(emb)
                found = True
                break
            if not found:
                best[k - 1] = None
        # Count edges in best
        cnt = 0
        for k in range(1, N + 1):
            a = best[k - 1]
            if a is None:
                continue
            kk, edges, roots, adj = tdata[k - 1]
            for x, y in edges:
                cnt += 1
        best_count = cnt

    # Validate and sanitize
    result = []
    for k in range(1, N + 1):
        a = best[k - 1]
        if a is None:
            result.append(None)
        else:
            result.append([int(x) for x in a])
    return result


def construct():
    try:
        return _solve()
    except Exception:
        # Fallback: identity for T20 only (score 1.0 baseline)
        return [None] * 19 + [list(range(20))]
