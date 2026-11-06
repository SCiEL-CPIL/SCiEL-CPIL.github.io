import time
from collections import deque

_START = time.time()
DEADLINE = _START + 130.0


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
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []

    def find_longest_cycle_from(start, max_depth=25):
        """Find a long cycle using DFS with path tracking and backtracking.
        Returns list of vertices in cycle order, or None."""
        best_cycle = None
        best_len = 0
        parent = {}
        visited = set()

        def dfs(u, depth):
            nonlocal best_cycle, best_len
            if time.time() > DEADLINE:
                return
            # Prune: if we already have a very good cycle and depth is small
            if best_cycle is not None and len(best_cycle) >= 15 and depth > 10:
                return
            if depth >= max_depth:
                return

            # Sort neighbors by degree (descending) to find longer cycles faster
            neighbors = sorted(rem[u], key=lambda x: -len(rem[x]))
            for v in neighbors:
                if v == parent.get(u, -1):
                    continue
                if v in visited:
                    # Found a cycle: reconstruct from v to u
                    cyc = [u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.reverse()
                    if len(cyc) > best_len:
                        best_len = len(cyc)
                        best_cycle = cyc[:]
                    return
                parent[v] = u
                visited.add(v)
                dfs(v, depth + 1)
                visited.discard(v)
                # Early exit if we found a very long cycle
                if best_cycle is not None and len(best_cycle) > 20:
                    return

        visited.add(start)
        parent[start] = -1
        dfs(start, 0)

        if best_cycle is not None:
            return best_cycle
        return None

    def find_cycle_any():
        """Find any cycle in the remaining graph using BFS."""
        for start in range(n):
            if not rem[start]:
                continue
            parent = {start: -1}
            visited = {start}
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in list(rem[u]):
                    if v == parent.get(u, -1):
                        continue
                    if v in visited:
                        # Found a cycle
                        cyc = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cyc.append(cur)
                        cyc.reverse()
                        if len(cyc) >= 3:
                            return cyc
                    parent[v] = u
                    visited.add(v)
                    queue.append(v)
        return None

    # Main loop: repeatedly find cycles and remove them
    # Strategy: try to find longer cycles by starting from high-degree vertices
    # and using a DFS that explores deeper paths with backtracking

    while time.time() < DEADLINE:
        # Get vertices with remaining edges, sorted by degree
        degs = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not degs:
            break
        degs.sort(reverse=True)

        found = None
        best_found_len = 0

        # Try several starting vertices, prefer high-degree ones
        for _, start in degs[:10]:
            if time.time() > DEADLINE:
                break
            cyc = find_longest_cycle_from(start)
            if cyc is not None and len(cyc) > best_found_len:
                best_found_len = len(cyc)
                found = cyc
            # Early exit if we found a very long cycle
            if best_found_len > 15:
                break

        if found is None:
            # Fall back to finding any cycle
            found = find_cycle_any()

        if found is None:
            break

        # Remove cycle edges from rem
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(cyc)

    # Leftover edges
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
        if time.time() > DEADLINE:
            # Fallback: just return edges as individual pieces
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    return results
