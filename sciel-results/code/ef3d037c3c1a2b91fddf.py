import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_long_cycle(adj, n, max_starts=30, max_restarts=8, rng=None):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy with improved scoring.
    Returns the cycle as a list of vertices or None.
    """
    if rng is None:
        rng = random.Random(42)

    best_cycle = None
    best_len = 0

    num_starts = min(n, max_starts)
    if n <= max_starts:
        starts = list(range(n))
    else:
        degrees = [len(adj[v]) for v in range(n)]
        starts = rng.sample(range(n), num_starts)

    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]

                if not unvisited:
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    if len(path) >= 4 and start in adj[current] and rng.random() < 0.3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    scored = []
                    for v in unvisited:
                        unvisited_neighbors = sum(1 for w in adj[v] if w not in visited)
                        score = unvisited_neighbors
                        if v == start:
                            score += 10
                        scored.append((score, v))

                    scored.sort(reverse=True)
                    top_k = min(3, len(scored))
                    if rng.random() < 0.7:
                        best_next = scored[0][1]
                    else:
                        best_next = rng.choice([s[1] for s in scored[:top_k]])

                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next

    return best_cycle


def _generate_cycle_pool(adj, n, max_cycles=200, rng=None, deadline=None):
    """
    Generate a pool of diverse cycles using randomized DFS.
    Returns a list of (cycle_edges_frozenset, vertex_list) tuples.
    """
    if rng is None:
        rng = random.Random(999)

    cycles = []
    seen = set()

    for attempt in range(5000):
        if deadline and time.time() > deadline:
            break
        if len(cycles) >= max_cycles:
            break

        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = rng.choice(candidates)

        path = [start]
        visited = {start}
        current = start

        for _ in range(100):
            neighbors = list(adj[current])
            unvisited = [v for v in neighbors if v not in visited]

            if not unvisited:
                if start in adj[current] and len(path) >= 3:
                    cycle_edges = frozenset(
                        tuple(sorted((path[i], path[(i + 1) % len(path)])))
                        for i in range(len(path))
                    )
                    if cycle_edges not in seen:
                        seen.add(cycle_edges)
                        cycles.append((cycle_edges, path[:]))
                break
            else:
                if len(path) >= 4 and start in adj[current] and rng.random() < 0.15:
                    cycle_edges = frozenset(
                        tuple(sorted((path[i], path[(i + 1) % len(path)])))
                        for i in range(len(path))
                    )
                    if cycle_edges not in seen:
                        seen.add(cycle_edges)
                        cycles.append((cycle_edges, path[:]))
                    break
                else:
                    next_v = rng.choice(unvisited)
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v

    return cycles


def _solve_ilp(edges, n, deadline):
    """
    Solve the decomposition as an ILP using scipy's linprog or a simple greedy ILP solver.
    Since we don't have a full ILP solver, we use a greedy set cover approach
    with the cycle pool.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    edge_set = set(tuple(sorted(e)) for e in edges)
    total_edges = len(edge_set)

    if total_edges == 0:
        return []

    # Generate cycle pool
    rng = random.Random(42)
    pool = _generate_cycle_pool(adj, n, max_cycles=300, rng=rng, deadline=deadline - 5)

    if not pool:
        return [[u, v] for u, v in edges]

    # Greedy set cover: repeatedly pick the cycle that covers the most uncovered edges
    uncovered = set(edge_set)
    pieces = []

    while uncovered:
        if time.time() > deadline:
            break

        best_cycle = None
        best_cover = 0

        for cycle_edges, vertex_list in pool:
            cover = len(cycle_edges & uncovered)
            if cover > best_cover:
                best_cover = cover
                best_cycle = (cycle_edges, vertex_list)

        if best_cycle is None or best_cover == 0:
            # No cycle covers any uncovered edge, add single edges
            for e in list(uncovered):
                pieces.append(list(e))
            break

        cycle_edges, vertex_list = best_cycle
        pieces.append(vertex_list[:])
        uncovered -= cycle_edges

    # Add any remaining uncovered edges as single edges
    for e in uncovered:
        pieces.append(list(e))

    return pieces


def _decompose_greedy(edges, n, deadline, rng=None):
    """
    Greedy decomposition: repeatedly find a cycle and remove it.
    """
    if rng is None:
        rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        best_cycle = _find_long_cycle(adj, n, max_starts=min(n, 25), max_restarts=6, rng=rng)

        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try ILP-based approach first
    if time.time() < deadline - 10:
        ilp_pieces = _solve_ilp(edges, n, deadline - 5)
        best_count = len(ilp_pieces)
        best = ilp_pieces

    # Try multiple random greedy strategies
    base_seed = 12345
    attempts = 0
    max_attempts = 100

    while time.time() < deadline - 1.0:
        seed = base_seed + attempts
        rng = random.Random(seed)
        pieces = _decompose_greedy(edges, n, deadline - 0.5, rng)
        cnt = len(pieces)

        if cnt < best_count:
            best_count = cnt
            best = pieces

        attempts += 1

        if attempts >= max_attempts:
            break

    return best if best else [[u, v] for u, v in edges]


def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
