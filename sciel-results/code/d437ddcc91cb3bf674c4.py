"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle(adj, start, deadline, max_nodes=2000000):
    """Find any simple cycle (length >= 3) in the current graph.

    The returned cycle visits only vertices that still have degree >= 2 in the
    current graph, which guarantees that removing its edges cannot strand any
    vertex at degree < 2 (a necessary condition for a Hamiltonian cycle to
    still exist). This keeps the search productive for packing many cycles.
    """
    n = len(adj)
    path = [start]
    visited = [False] * n
    visited[start] = True
    nodes = [0]

    def dfs():
        if time.time() > deadline:
            return False
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return False
        cur = path[-1]
        # a vertex can only be part of a cycle if it has at least 2 remaining edges
        if len(adj[cur]) < 2:
            return False
        neighbors = [v for v in adj[cur] if not visited[v] and len(adj[v]) >= 2]
        if not neighbors:
            return False
        # Warnsdorff: prefer the most constrained (fewest onward options) first
        neighbors.sort(key=lambda v: len([w for w in adj[v] if not visited[w] and len(adj[w]) >= 2]))
        for v in neighbors:
            if time.time() > deadline:
                return False
            path.append(v)
            visited[v] = True
            if dfs():
                return True
            path.pop()
            visited[v] = False
        return False

    if dfs():
        return list(path)
    return None

def _polish_cycle(cycle, adj, deadline):
    """Try to shorten a Hamiltonian cycle via 2-opt edge swaps, freeing edges.

    Returns a shorter cycle (a subset of the original vertex set) if a valid
    shorter cycle is found, otherwise returns the original cycle.
    """
    n = len(cycle)
    if n < 4:
        return cycle
    pos = {v: i for i, v in enumerate(cycle)}
    edges = set()
    for i in range(n):
        edges.add(frozenset((cycle[i], cycle[(i + 1) % n])))

    best = list(cycle)
    for _ in range(60):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best)):
            a = best[i]
            b = best[(i + 1) % len(best)]
            for j in range(i + 2, len(best)):
                if j == (i + 1) % len(best):
                    continue
                c = best[j]
                d = best[(j + 1) % len(best)]
                if frozenset((a, b)) in edges or frozenset((c, d)) in edges:
                    continue
                if frozenset((a, c)) in edges or frozenset((b, d)) in edges:
                    continue
                if frozenset((a, c)) not in adj[a] or frozenset((b, d)) not in adj[b]:
                    continue
                # 2-opt: reverse the segment between b and c
                seg = best[i + 1:j + 1][::-1]
                newc = best[:i + 1] + seg + best[j + 1:]
                # verify the two new closing edges exist
                if newc[-1] not in adj[newc[0]]:
                    continue
                new_edges = set()
                for k in range(len(newc)):
                    e = frozenset((newc[k], newc[(k + 1) % len(newc)]))
                    if e in new_edges:
                        continue
                    if e not in adj[list(e)[0]]:
                        break
                    new_edges.add(e)
                else:
                    best = newc
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best

def _valid_result(cycles, n, edgeset):
    if not cycles or len(cycles) > 10:
        return False
    used = set()
    for c in cycles:
        if not isinstance(c, (list, tuple)):
            return False
        if len(c) < 3:
            return False
        if any(not isinstance(v, int) or isinstance(v, bool) for v in c):
            return False
        if len(set(c)) != len(c):
            return False
        if set(c) != set(range(n)):
            return False
        m = len(c)
        for i in range(m):
            e = frozenset((c[i], c[(i + 1) % m]))
            if e not in edgeset:
                return False
            if e in used:
                return False
            used.add(e)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin for the 90 s wall-clock limit
    n = P['n']
    base_adj = _build_adj(n, P['edges'])
    edgeset = set(frozenset(e) for e in P['edges'])
    full = set(range(n))

    best = []

    def consider(cycles):
        nonlocal best
        if len(cycles) > len(best) and _valid_result(cycles, n, edgeset):
            best = [list(c) for c in cycles]

    # Strategy A: pack edge-disjoint cycles greedily (Hamiltonian-first, then
    # any remaining cycle). This generalises the seed and can reach 4 cycles
    # by using the 7 spare edges for a shorter final cycle.
    for restart in range(30):
        if time.time() > deadline:
            break
        adj = [set(s) for s in base_adj]
        cycles = []
        for _ in range(10):
            if time.time() > deadline:
                break
            degs = [len(adj[v]) for v in range(n)]
            if sum(1 for d in degs if d < 2) > 0:
                break
            # prefer high-degree starts
            starts = sorted(range(n), key=lambda v: -degs[v])
            found = None
            for s in starts:
                if time.time() > deadline:
                    break
                c = _find_cycle(adj, s, deadline)
                if c is not None:
                    found = c
                    break
            if found is None:
                break
            for i in range(len(found)):
                u = found[i]
                v = found[(i + 1) % len(found)]
                adj[u].discard(v)
                adj[v].discard(u)
            cycles.append(found)
            consider(cycles)
            if len(cycles) >= 10:
                break

    # Strategy B: three Hamiltonian cycles + 2-opt polish + a 4th cycle.
    # The polish shortens a Hamiltonian cycle to free edges for an extra one.
    for restart in range(20):
        if time.time() > deadline:
            break
        adj = [set(s) for s in base_adj]
        cycles = []
        for _ in range(3):
            if time.time() > deadline:
                break
            starts = sorted(range(n), key=lambda v: -len(adj[v]))
            found = None
            for s in starts:
                if time.time() > deadline:
                    break
                c = _find_cycle(adj, s, deadline)
                if c is not None and len(c) == n:
                    found = c
                    break
            if found is None:
                break
            for i in range(len(found)):
                u = found[i]
                v = found[(i + 1) % len(found)]
                adj[u].discard(v)
                adj[v].discard(u)
            cycles.append(found)
        if len(cycles) == 3:
            # polish one cycle to free edges, then try to add a 4th
            adj = [set(s) for s in base_adj]
            for i in range(len(cycles)):
                for k in range(len(cycles[i])):
                    u = cycles[i][k]
                    v = cycles[i][(k + 1) % len(cycles[i])]
                    adj[u].discard(v)
                    adj[v].discard(u)
            polished = _polish_cycle(cycles[0], base_adj, deadline)
            if len(polished) < n:
                adj2 = [set(s) for s in base_adj]
                for i in range(len(cycles)):
                    cc = polished if i == 0 else cycles[i]
                    for k in range(len(cc)):
                        u = cc[k]
                        v = cc[(k + 1) % len(cc)]
                        adj2[u].discard(v)
                        adj2[v].discard(u)
                cands = [polished] + cycles[1:]
                if _valid_result(cands, n, edgeset):
                    consider(cands)
                starts = sorted(range(n), key=lambda v: -len(adj2[v]))
                for s in starts:
                    if time.time() > deadline:
                        break
                    c4 = _find_cycle(adj2, s, deadline)
                    if c4 is not None:
                        cands2 = [polished] + cycles[1:] + [c4]
                        consider(cands2)
                        break

    # Fallback: guarantee a valid nonempty result (a single Hamiltonian cycle).
    if not best:
        fallback = [list(range(n))]
        if _valid_result(fallback, n, edgeset):
            best = fallback
        else:
            # extremely defensive: search for any valid cycle
            adj = [set(s) for s in base_adj]
            for s in range(n):
                c = _find_cycle(adj, s, deadline)
                if c is not None and _valid_result([c], n, edgeset):
                    best = [list(c)]
                    break

    result = [[int(v) for v in c] for c in best]
    return result
# EVOLVE-BLOCK-END
