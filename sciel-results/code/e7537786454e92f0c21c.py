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


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle(adj, deadline):
    """
    Find a long cycle in the graph. Uses a randomized DFS approach with
    backtracking to find a simple cycle. Tries multiple starting points.
    """
    n = len(adj)
    if n == 0:
        return None

    # Try to find a cycle using randomized DFS
    best_cycle = None
    start_candidates = list(adj.keys())
    random.shuffle(start_candidates)

    for start in start_candidates[:min(20, len(start_candidates))]:
        if time.time() > deadline:
            break
        # DFS to find a cycle back to start
        stack = [(start, [start], {start})]
        found = None
        while stack and not found:
            if time.time() > deadline:
                break
            node, path, visited = stack.pop()
            neighbors = list(adj[node])
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    found = path
                    break
                if nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))
        if found and (best_cycle is None or len(found) > len(best_cycle)):
            best_cycle = found

    return best_cycle


def _find_cycle_bfs(adj, deadline):
    """
    Find a cycle by looking for a path from a node to one of its neighbors
    that doesn't use the direct edge. This guarantees a cycle of length >= 3.
    """
    n = len(adj)
    if n == 0:
        return None

    best_cycle = None
    nodes = list(adj.keys())
    random.shuffle(nodes)

    for start in nodes[:min(30, len(nodes))]:
        if time.time() > deadline:
            break
        # For each neighbor, try to find a path from start to neighbor
        # that doesn't use the direct edge (i.e., path length >= 2)
        neighbors = list(adj[start])
        random.shuffle(neighbors)
        for end in neighbors:
            if time.time() > deadline:
                break
            # BFS from start to end, not using the direct edge
            parent = {start: None}
            queue = deque([start])
            found = False
            while queue and not found:
                if time.time() > deadline:
                    break
                node = queue.popleft()
                if node == end and parent.get(end) != start:
                    # Reconstruct path
                    path = []
                    cur = end
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    if len(path) >= 3:
                        if best_cycle is None or len(path) > len(best_cycle):
                            best_cycle = path
                    found = True
                    break
                for nb in adj[node]:
                    if nb not in parent:
                        # Don't use the direct edge from start to end
                        if node == start and nb == end:
                            continue
                        parent[nb] = node
                        queue.append(nb)

    return best_cycle


def _decompose(edges, time_limit):
    """
    Decompose edges into cycles and single edges.
    Greedy: repeatedly find a cycle, remove its edges, repeat.
    Remaining edges are single-edge pieces.
    """
    if not edges:
        return []

    deadline = time.time() + time_limit

    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []

    while remaining:
        if time.time() > deadline:
            # Time's up: all remaining edges as single pieces
            for e in remaining:
                pieces.append(list(e))
            break

        adj = _build_adj(remaining)

        # Try to find a cycle
        cycle = None

        # Strategy 1: BFS-based cycle finding
        if time.time() < deadline:
            cycle = _find_cycle_bfs(adj, deadline)

        # Strategy 2: DFS-based cycle finding
        if cycle is None and time.time() < deadline:
            cycle = _find_cycle(adj, deadline)

        if cycle and len(cycle) >= 3:
            # Remove cycle edges from remaining
            cycle_edges = set()
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    cycle_edges.add(edge)
                else:
                    # This shouldn't happen if the cycle is valid
                    cycle = None
                    break

            if cycle:
                remaining -= cycle_edges
                pieces.append(cycle)
            else:
                # Cycle wasn't valid; fall through to single edge
                pass

        if not cycle or len(cycle) < 3:
            # No cycle found; take a single edge
            # Prefer edges that are "harder" to include in cycles (low degree endpoints)
            if remaining:
                # Pick an edge whose removal leaves the graph less connected
                best_edge = None
                best_score = float('inf')
                for e in remaining:
                    u, v = e
                    deg_u = len(adj.get(u, set()))
                    deg_v = len(adj.get(v, set()))
                    score = deg_u + deg_v
                    if score < best_score:
                        best_score = score
                        best_edge = e
                if best_edge is None:
                    best_edge = next(iter(remaining))
                remaining.discard(tuple(sorted(best_edge)))
                pieces.append(list(best_edge))

    return pieces


def construct():
    total_deadline = time.time() + 140  # Safety margin before 150s kill

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for i, spec in enumerate(specs):
        # Allocate time proportionally; give more to larger graphs
        elapsed = time.time() - (total_deadline - 140)
        remaining_time = max(1.0, 140 - elapsed - 5)  # 5s safety
        per_graph = remaining_time / (len(specs) - i)

        edges = _instance(*spec)
        pieces = _decompose(edges, per_graph)
        results.append(pieces)

    return results

# EVOLVE-BLOCK-END
