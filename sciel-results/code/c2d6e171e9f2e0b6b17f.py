"""Decompose fixed instance graphs into few cycles + edges (Erdős-Gallai style)."""
import time
from collections import defaultdict, deque
from random import Random

# EVOLVE-BLOCK-START
def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, time_limit=10.0):
    """
    Decompose edges into edge-disjoint cycles (maximizing cycle length) and leftover edges.
    Strategy:
    1. Repeatedly find the longest cycle in the current graph (via DFS from each vertex).
    2. Remove its edges.
    3. When no cycles remain, each remaining edge is a single-edge piece.
    For small n (<=30), exhaustive DFS for longest cycle is feasible within time limits.
    """
    if not edges:
        return []

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    start_time = time.time()
    deadline = start_time + time_limit

    def find_longest_cycle(adj, n):
        """
        Find the longest simple cycle in the graph represented by adj.
        Uses DFS from each vertex, tracking the current path.
        Returns list of vertices in the cycle (without repeating the start at the end),
        or None if no cycle exists.
        """
        best_cycle = None
        best_len = 0

        for start in range(n):
            if len(adj[start]) < 2:
                continue

            # DFS to find all simple cycles starting from 'start'
            # We only consider cycles where 'start' is the smallest vertex to avoid duplicates
            stack = [(start, [start], {start})]
            while stack:
                u, path, visited = stack.pop()
                # Check if we can close the cycle back to start
                if len(path) >= 3 and start in adj[u]:
                    cycle_len = len(path)
                    if cycle_len > best_len:
                        best_len = cycle_len
                        best_cycle = path[:]

                for v in adj[u]:
                    if v == start:
                        # Only close if path length >= 3 (at least 3 distinct vertices)
                        if len(path) >= 3:
                            cycle_len = len(path)
                            if cycle_len > best_len:
                                best_len = cycle_len
                                best_cycle = path[:]
                        continue
                    if v in visited:
                        continue
                    # To avoid duplicate cycles, only extend if v > start
                    # Actually, standard approach: just avoid revisiting nodes in current path
                    # To reduce search space, we can impose an ordering: only visit nodes > start
                    # But this might miss some cycles. Let's just do standard DFS with visited set.
                    # For n<=30, this is okay if we prune well.
                    # Optimization: only consider v > start to find each cycle exactly once?
                    # No, that's for counting. For finding the longest, we can just DFS.
                    # But to speed up, let's try to find long paths first.
                    
                    # Heuristic: visit neighbors in order of degree (higher degree first might lead to longer cycles?)
                    # Or just standard.
                    new_path = path + [v]
                    new_visited = visited | {v}
                    stack.append((v, new_path, new_visited))

        return best_cycle

    # Greedily extract longest cycles
    while time.time() < deadline:
        cycle = find_longest_cycle(adj, n)
        if cycle is None:
            break

        # Add cycle to pieces
        pieces.append([int(v) for v in cycle])

        # Remove edges of the cycle from adj
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            adj[u].discard(v)
            adj[v].discard(u)

    # Add remaining edges as single-edge pieces
    remaining_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                remaining_edges.add((u, v))

    for u, v in sorted(remaining_edges):
        pieces.append([u, v])

    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    start_time = time.time()
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        # Total limit 150s, use 145s for computation to be safe
        remaining_time = 145.0 - elapsed
        graphs_left = len(specs) - i
        if graphs_left <= 0:
            break
            
        alloc_time = remaining_time / graphs_left
        # Cap per-graph time to avoid one graph taking too long, but ensure at least 1s
        alloc_time = max(1.0, min(alloc_time, 40.0))

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=alloc_time)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
