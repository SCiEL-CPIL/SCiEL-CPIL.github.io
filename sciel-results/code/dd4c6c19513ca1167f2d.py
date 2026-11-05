import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_longest_cycle_heuristic(adj, n, edge_set, time_limit=2.0, rng=None):
    """
    Finds a long simple cycle in the graph using randomized DFS with backtracking.
    Returns the cycle as a list of vertices, or None if no cycle found.
    """
    if rng is None:
        rng = random.Random(42)
        
    start_time = time.time()
    best_cycle = None
    best_len = 0
    seen_canonical = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_val = min(cycle)
        idxs = [i for i, v in enumerate(cycle) if v == min_val]
        candidates = []
        for idx in idxs:
            c1 = cycle[idx:] + cycle[:idx]
            c2 = list(reversed(c1))
            candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
        return min(candidates)

    def _is_valid_cycle(cycle):
        if len(cycle) < 3:
            return False
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return False
        return True

    # Try multiple random starts
    num_trials = 500
    for _ in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        start = rng.randrange(n)
        if not adj[start]:
            continue
            
        visited = [False] * n
        path = []
        
        # DFS
        stack = [(start, iter(adj[start]))]
        visited[start] = True
        path.append(start)
        
        while stack:
            if time.time() - start_time > time_limit:
                break
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    key = _canonical(cycle)
                    if key and key not in seen_canonical:
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle
                        seen_canonical.add(key)
                    break # Backtrack after finding a cycle? Or continue? 
                    # To find *longer* cycles, we should continue searching.
                    # But breaking here ensures we don't loop forever on this path.
                    # Actually, if we break, we pop the stack.
                    # We want to find the LONGEST. So we should continue exploring other neighbors.
                    # However, the standard DFS for longest path is hard.
                    # Let's just record and continue.
                elif not visited[nb]:
                    visited[nb] = True
                    path.append(nb)
                    stack.append((nb, iter(adj[nb])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    path.pop()
                    visited[node] = False

    return best_cycle


def _solve_one(edges, n, deadline):
    if not edges:
        return []

    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edge_set, n)

    if m < 3:
        return [list(e) for e in edges]

    pieces = []
    rng = random.Random(12345)
    
    # Greedy Longest Cycle Extraction
    while time.time() < deadline - 1.0:
        # Find the longest cycle in the current graph
        cycle = _find_longest_cycle_heuristic(adj, n, edge_set, time_limit=1.0, rng=rng)
        
        if not cycle:
            break
        
        # Extract edges of the cycle
        cycle_edges = set()
        L = len(cycle)
        for i in range(L):
            u = cycle[i]
            v = cycle[(i + 1) % L]
            e = tuple(sorted((u, v)))
            cycle_edges.add(e)
            
        # Remove from graph
        edge_set -= cycle_edges
        for u, v in cycle_edges:
            adj[u].remove(v)
            adj[v].remove(u)
            
        pieces.append(cycle)
        
        # If we removed all edges, we are done
        if not edge_set:
            break

    # Remaining edges are single pieces
    for e in edge_set:
        pieces.append(list(e))
        
    return pieces


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
