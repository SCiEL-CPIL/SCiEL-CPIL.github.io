"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
import time
import random
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, deadline):
    """
    Decompose edges into cycles and single edges.
    Repeatedly find long cycles (prefer high-degree vertices), remove them,
    then output leftover edges as single pieces.
    """
    if not edges:
        return []

    # Build adjacency as dict of sets for fast updates
    rem = {i: set() for i in range(n)}
    for u, v in edges:
        rem[u].add(v)
        rem[v].add(u)

    pieces = []

    def find_cycle_dfs(start):
        """
        DFS from start to find a cycle. Explores high-degree neighbors first
        to find longer cycles. Returns cycle as list of vertices or None.
        """
        best_cycle = None
        best_len = 0

        # Iterative DFS with explicit stack to avoid recursion depth issues
        # Each stack entry: (node, parent, path, visited_set)
        # We use a visited set per path to track the current path

        # For efficiency, we do a bounded DFS
        stack = [(start, -1, [start], {start})]
        visited_global = set()  # Not used for pruning across branches, just for tracking

        while stack:
            if time.time() > deadline:
                break
            node, parent, path, visited = stack.pop()

            # Try neighbors in order of decreasing degree (to find longer cycles)
            neighbors = list(rem[node])
            neighbors.sort(key=lambda x: -len(rem[x]))

            for nb in neighbors:
                if time.time() > deadline:
                    break
                if nb == parent:
                    continue
                if nb == start and len(path) >= 3:
                    # Found a cycle back to start
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    break
                if nb not in visited:
                    new_path = path + [nb]
                    new_visited = visited | {nb}
                    stack.append((nb, node, new_path, new_visited))

        return best_cycle

    def find_cycle_bfs():
        """
        Find any cycle using BFS. Returns cycle as list of vertices or None.
        """
        for start in range(n):
            if not rem[start]:
                continue
            parent = {start: -1}
            visited = {start}
            queue = deque([start])
            while queue:
                if time.time() > deadline:
                    return None
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
                    else:
                        parent[v] = u
                        visited.add(v)
                        queue.append(v)
        return None

    # Main loop: repeatedly find cycles and remove them
    while time.time() < deadline:
        # Get vertices with remaining edges, sorted by degree descending
        active = [(len(rem[i]), i) for i in range(n) if rem[i]]
        if not active:
            break
        active.sort(reverse=True)

        found = None
        best_found_len = 0

        # Try several starting vertices, prefer high-degree ones
        for deg, start in active[:min(10, len(active))]:
            if time.time() > deadline:
                break
            cyc = find_cycle_dfs(start)
            if cyc is not None and len(cyc) > best_found_len:
                best_found_len = len(cyc)
                found = cyc
            # Early exit if we found a very long cycle
            if best_found_len > 12:
                break

        # Fallback to BFS if DFS didn't find anything
        if found is None:
            found = find_cycle_bfs()

        if found is None:
            # No more cycles; break out
            break

        # Remove cycle edges from rem
        for i in range(len(found)):
            a, b = found[i], found[(i + 1) % len(found)]
            rem[a].discard(b)
            rem[b].discard(a)
        pieces.append(found)

    # Leftover edges as single pieces
    leftover = set()
    for u in range(n):
        for v in rem[u]:
            if u < v:
                leftover.add((u, v))
    for u, v in sorted(leftover):
        pieces.append([u, v])

    return pieces


def construct():
    start_time = time.time()
    # Leave 10 seconds buffer before the 150s kill
    deadline = start_time + 140.0

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        if time.time() > deadline - 5.0:
            # Not enough time left; return empty pieces for remaining graphs
            # This will be invalid, but better than timeout
            results.append([])
            continue

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)

    return results

# EVOLVE-BLOCK-END
