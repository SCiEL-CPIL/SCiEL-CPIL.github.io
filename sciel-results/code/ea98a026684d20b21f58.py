"""Decompose each fixed instance graph into edge-disjoint cycles + single edges."""
import time
import random
from random import Random
import sys


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_small_cycles(adj, max_len=4):
    cycles = []
    if not adj:
        return cycles
    n = max(adj.keys()) + 1
    mat = [[False] * n for _ in range(n)]
    for u, nbrs in adj.items():
        for v in nbrs:
            mat[u][v] = True
            mat[v][u] = True

    # 3-cycles
    for u in range(n):
        for v in range(u + 1, n):
            if mat[u][v]:
                for w in range(v + 1, n):
                    if mat[v][w] and mat[w][u]:
                        cycles.append([u, v, w])

    # 4-cycles
    for u in range(n):
        for v in range(u + 1, n):
            if not mat[u][v]:
                continue
            for w in range(v + 1, n):
                if not mat[v][w]:
                    continue
                for x in range(w + 1, n):
                    if mat[w][x] and mat[x][u] and not mat[u][w] and not mat[v][x]:
                        cycles.append([u, v, w, x])

    return cycles


def _decompose(edges, deadline):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    avail = set(tuple(sorted(e)) for e in edges)
    pieces = []

    # Pre-find small cycles
    small_cycles = _find_small_cycles(adj, 4)
    # Sort by length descending
    small_cycles.sort(key=lambda c: len(c), reverse=True)
    
    for c in small_cycles:
        if time.time() > deadline:
            break
        # Check if all edges in c are available
        valid = True
        for i in range(len(c)):
            a, b = c[i], c[(i + 1) % len(c)]
            e = (min(a, b), max(a, b))
            if e not in avail:
                valid = False
                break
        if valid:
            for i in range(len(c)):
                a, b = c[i], c[(i + 1) % len(c)]
                e = (min(a, b), max(a, b))
                avail.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
            pieces.append(c)

    def try_greedy_cycle(start, max_attempts=3):
        """Try to extend from start to form a cycle back to start. Dry run."""
        for _ in range(max_attempts):
            path = [start]
            visited = {start}
            cur = start
            for _ in range(len(adj) * 2):
                if time.time() > deadline:
                    return None
                nbrs = [nb for nb in adj[cur] if (min(cur, nb), max(cur, nb)) in avail]
                if not nbrs:
                    return None
                if len(path) >= 3 and start in nbrs:
                    return path[:]
                unvisited = [nb for nb in nbrs if nb not in visited]
                if unvisited:
                    weights = [len(adj[nb]) + 1 for nb in unvisited]
                    total = sum(weights)
                    r = random.random() * total
                    cum = 0
                    chosen = unvisited[0]
                    for nb, w in zip(unvisited, weights):
                        cum += w
                        if r <= cum:
                            chosen = nb
                            break
                else:
                    if start in nbrs and len(path) >= 3:
                        return path[:]
                    return None
                path.append(chosen)
                visited.add(chosen)
                cur = chosen
        return None

    while time.time() < deadline:
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break

        best_cycle = None
        best_len = 0

        rng_local = Random()
        num_starts = min(len(candidates), 20)
        starts = rng_local.sample(candidates, num_starts) if num_starts < len(candidates) else candidates

        for s in starts:
            if time.time() > deadline:
                break
            cycle = try_greedy_cycle(s, max_attempts=5)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle

        if not best_cycle:
            for s in starts[:5]:
                if time.time() > deadline:
                    break
                cycle = try_greedy_cycle(s, max_attempts=10)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle

        if not best_cycle:
            break

        # Commit the best cycle found
        for i in range(len(best_cycle)):
            a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = (min(a, b), max(a, b))
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        pieces.append(best_cycle)

    for e in sorted(avail):
        pieces.append(list(e))

    return pieces


def construct():
    deadline = time.time() + 140.0
    random.seed(42)
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    total_size = sum(n * n for n, _, _ in specs)
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            results.append([list(e) for e in edges])
            continue
        frac = (n * n) / total_size
        graph_deadline = time.time() + max(5.0, remaining_time * frac * 0.9)
        
        # Run multiple times with different random seeds to find best decomposition
        best_pieces = None
        best_count = float('inf')
        
        # Use a fixed number of restarts based on time available
        # Each restart is fast for small n
        restarts = 10
        
        for r in range(restarts):
            if time.time() > graph_deadline:
                break
            random.seed(42 + r)
            pieces = _decompose(edges, graph_deadline)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)
    return results
