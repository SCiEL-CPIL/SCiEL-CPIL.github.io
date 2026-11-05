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
    Uses a randomized greedy strategy with better heuristics.
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
        # Prioritize high-degree vertices
        starts = sorted(range(n), key=lambda v: degrees[v], reverse=True)[:num_starts]
        rng.shuffle(starts)

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
                    # If path is long enough and we can close, consider closing
                    if len(path) >= 5 and start in adj[current] and rng.random() < 0.2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    # Score neighbors: prefer those with more unvisited connections
                    scored = []
                    for v in unvisited:
                        unvisited_neighbors = sum(1 for w in adj[v] if w not in visited)
                        score = unvisited_neighbors * 2
                        # Slight preference for vertices closer to start (if we can detect it)
                        if v == start:
                            score += 10
                        scored.append((score, v))

                    scored.sort(reverse=True)
                    top_k = min(3, len(scored))
                    if rng.random() < 0.6:
                        best_next = scored[0][1]
                    else:
                        best_next = rng.choice([s[1] for s in scored[:top_k]])

                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next

    return best_cycle


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

    base_seed = 12345
    attempts = 0
    max_attempts = 200

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
