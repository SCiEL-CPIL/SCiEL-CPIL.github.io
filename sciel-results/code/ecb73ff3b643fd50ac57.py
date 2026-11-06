import time

_START = time.time()
DEADLINE = _START + 135.0

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n):
    """Decompose edge set into edge-disjoint cycles + leftover single edges,
    minimizing piece count. A cycle of length L replaces L single-edge pieces
    with 1, so longer cycles are strictly better."""
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # remaining adjacency + degree bookkeeping
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []
    # iterate: find a cycle in the remaining graph, remove it, repeat
    # use a heuristic: repeatedly extract cycles via DFS path-finding
    # We look for a cycle by finding any edge (u,v) where u,v are connected
    # in the graph after removing that edge.

    def find_cycle():
        # Find any cycle in rem. Use DFS to find a back edge.
        # Return list of vertices in cycle order, or None.
        for start in range(n):
            if not rem[start]:
                continue
            # DFS
            stack = [(start, -1, [start])]
            # We need to detect a cycle. Use standard DFS with parent tracking.
            parent = {start: -1}
            visited = {start}
            # iterative DFS
            def dfs(u):
                for v in list(rem[u]):
                    if v == parent.get(u, -1):
                        continue
                    if v in visited:
                        # found a cycle: path from v to u plus edge u-v
                        # reconstruct path
                        cyc = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cyc.append(cur)
                        cyc.reverse()
                        return cyc
                    parent[v] = u
                    visited.add(v)
                    res = dfs(v)
                    if res is not None:
                        return res
                return None
            cyc = dfs(start)
            if cyc is not None:
                return cyc
        return None

    # Prefer longer cycles: do a few local improvements.
    # Strategy: repeatedly find a cycle; to get longer ones, we can try
    # to find a cycle through the highest-degree vertex first.
    import sys
    sys.setrecursionlimit(10000)

    while time.time() < DEADLINE:
        # pick a start vertex with high degree to encourage longer cycles
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)
        found = None
        for _, start in degs[:6]:  # try a few high-degree starts
            # DFS for a cycle
            parent = {start: -1}
            visited = {start}
            cyc = None
            # iterative to avoid recursion limits
            stack = [(start, iter(list(rem[start])))]
            while stack and cyc is None:
                u, it = stack[-1]
                advanced = False
                for v in it:
                    if v == parent.get(u, -1):
                        continue
                    if v in visited:
                        # cycle
                        cyc = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cyc.append(cur)
                        cyc.reverse()
                        break
                    parent[v] = u
                    visited.add(v)
                    stack.append((v, iter(list(rem[v]))))
                    advanced = True
                    break
                if not advanced:
                    stack.pop()
            if cyc is not None:
                found = cyc
                break
        if found is None:
            break
        # remove cycle edges
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(cyc)

    # leftover edges
    leftover = set()
    for u in range(n):
        for v in rem[u]:
            if u < v:
                leftover.add((u, v))
    for u, v in sorted(leftover):
        pieces.append([u, v])
    return pieces


def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    return results
