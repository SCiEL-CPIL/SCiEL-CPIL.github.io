"""Dense odd-order graph avoiding equal-degree path endpoints.

Strategy: start from K_{7,8} (provably valid, 56 edges), then greedily add
missing edges (safe when the two endpoints currently have different degrees)
and, when stuck, perform safe 1-edge rewires to free up a valid addition.
A deadline guard returns the best valid solution found before the limit.
"""
# EVOLVE-BLOCK-START
import time

_N = 15
_DEADLINE = 78.0  # wall-clock safety margin under the 90s kill
_START = time.time()


def _is_valid(adj, deg):
    """Full validity check: no simple 3-edge path has equal-degree endpoints."""
    n = _N
    # For each edge (u,v), every s in N(u)\{v} and t in N(v)\{u}, s!=t must
    # satisfy deg(s) != deg(t).
    for u in range(n):
        nu = adj[u]
        for v in nu:
            nv = adj[v]
            # endpoints from u-side (excluding v)
            for s in nu:
                if s == v:
                    continue
                ds = deg[s]
                for t in nv:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True


def _try_add(adj, deg, u, v):
    """Add edge (u,v) if absent and safe. Returns True if added."""
    if u == v:
        return False
    if v in adj[u]:
        return False
    if deg[u] == deg[v]:
        return False
    # add
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1
    return True


def _remove(adj, deg, u, v):
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1


def _edges_from(adj):
    out = []
    for u in range(_N):
        for v in adj[u]:
            if u < v:
                out.append([int(u), int(v)])
    return out


def construct():
    n = _N
    # Bipartition sizes 7 and 8 -> K_{7,8}, 56 edges, provably valid.
    A = list(range(7))
    B = list(range(7, 15))
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u in A:
        for v in B:
            adj[u].add(v)
            adj[v].add(u)
    for i in range(n):
        deg[i] = len(adj[i])

    best_edges = _edges_from(adj)
    best_count = len(best_edges)

    # Greedy addition pass (repeated until no safe add exists).
    def greedy_add():
        changed = True
        while changed:
            changed = False
            for u in range(n):
                if time.time() > _DEADLINE:
                    return
                for v in range(u + 1, n):
                    if _try_add(adj, deg, u, v):
                        changed = True
                        if len([1 for i in range(n) for j in adj[i] if i < j]) > best_count:
                            best_count = len([1 for i in range(n) for j in adj[i] if i < j])
                            best_edges = _edges_from(adj)
                        break  # degrees changed; restart scan

    greedy_add()

    # Rewire phase: try to free a valid addition by moving an edge.
    def rewire_pass():
        nonlocal best_edges, best_count
        progressed = True
        while progressed:
            progressed = False
            if time.time() > _DEADLINE:
                return
            # Try moving each edge (u,v) to (u,w) for w not adjacent to u.
            for u in range(n):
                if time.time() > _DEADLINE:
                    return
                for v in list(adj[u]):
                    if v <= u:
                        continue
                    # candidate new partner w
                    for w in range(n):
                        if w == u or w == v or w in adj[u]:
                            continue
                        # proposed: remove (u,v), add (u,w)
                        # safe if new endpoint degrees differ: deg(u) (unchanged) vs deg(w)
                        # and old edge removal must not create violation.
                        du = deg[u]
                        dw = deg[w]
                        if du == dw:
                            continue
                        # Simulate
                        _remove(adj, deg, u, v)
                        _try_add(adj, deg, u, w)
                        # Check: after removal, is graph still valid? and after add?
                        # We add safely (du != dw). Need removal to be safe.
                        # Removal of (u,v): previously safe adds remain; removal can't
                        # create a new equal-degree path (fewer paths). So valid.
                        if _is_valid(adj, deg):
                            progressed = True
                            cnt = len([1 for i in range(n) for j in adj[i] if i < j])
                            if cnt > best_count:
                                best_count = cnt
                                best_edges = _edges_from(adj)
                            # After a rewire, run greedy to add more
                            greedy_add()
                            break
                        else:
                            # revert
                            _remove(adj, deg, u, w)
                            _try_add(adj, deg, u, v)
                    else:
                        continue
                if progressed:
                    break

    rewire_pass()

    # Final validity guarantee
    if not _is_valid(adj, deg):
        # fall back to best recorded valid solution
        adj = [set() for _ in range(n)]
        for e in best_edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        return best_edges

    return _edges_from(adj)
# EVOLVE-BLOCK-END
