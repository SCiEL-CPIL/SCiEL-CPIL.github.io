"""Rewritten program: greedy longest-cycle extraction for edge decomposition."""
import time
import random
import sys
from collections import defaultdict

# Hard deadline: finish well before the 150s wall-clock kill.
DEADLINE = 120.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edge_set):
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle(adj, remaining, rng, max_tries=200, max_depth=400):
    """Find a simple cycle in the subgraph induced by `remaining` edges.

    Returns a list of vertices [v0, v1, ..., vk] (v0 != vk, closing edge implied)
    or None. Uses randomized DFS with backtracking to grow a path that closes.
    """
    if not remaining:
        return None

    # Candidate start vertices: prefer higher degree.
    starts = list(adj.keys())
    if not starts:
        return None

    for _ in range(max_tries):
        start = rng.choice(starts)
        # Randomize neighbor order for diversity.
        visited = {start}
        stack = [(start, None, list(adj[start]))]
        path = [start]

        # DFS to find a path that returns to start.
        found = None
        while stack and found is None:
            node, parent, neighbors = stack[-1]
            # Shuffle neighbors each time we revisit for randomness.
            rng.shuffle(neighbors)
            if not neighbors:
                stack.pop()
                path.pop()
                visited.discard(node)
                continue
            nxt = neighbors.pop()
            if nxt == start and len(path) >= 3:
                # Closing the cycle.
                found = list(path)
                break
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                stack.append((nxt, node, list(adj[nxt])))
        if found is not None:
            return found
    return None


def _decompose_once(edges, rng, deadline):
    """Greedy: repeatedly extract a cycle, remove its edges, leftovers are edges."""
    remaining = set(edges)
    pieces = []
    while remaining:
        if _elapsed() > deadline:
            # Fall back: remaining edges as single pieces.
            pieces.extend([list(e) for e in remaining])
            return pieces
        adj = _build_adj(remaining)
        cycle = _find_cycle(adj, remaining, rng)
        if cycle is None or len(cycle) < 3:
            # No cycle found; dump remaining as edges.
            pieces.extend([list(e) for e in remaining])
            return pieces
        # Record the cycle piece.
        pieces.append(list(cycle))
        # Remove the cycle's edges from remaining.
        k = len(cycle)
        for i in range(k):
            a = cycle[i]
            b = cycle[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
    return pieces


def _validate(pieces, edges):
    """Check that pieces are edge-disjoint and cover all edges exactly once."""
    edge_set = set(edges)
    seen = set()
    for p in pieces:
        if len(p) == 2:
            a, b = p
            e = (a, b) if a < b else (b, a)
            if e in seen or e not in edge_set:
                return False
            seen.add(e)
        elif len(p) >= 3:
            # Cycle: check distinct vertices.
            if len(set(p)) != len(p):
                return False
            k = len(p)
            for i in range(k):
                a = p[i]
                b = p[(i + 1) % k]
                e = (a, b) if a < b else (b, a)
                if e in seen or e not in edge_set:
                    return False
                seen.add(e)
        else:
            return False
    return seen == edge_set


def _decompose_graph(edges, deadline, rng):
    """Run multiple restarts, return best (fewest pieces) valid decomposition."""
    best = None
    best_count = float('inf')
    # Greedy is deterministic in structure but randomized in cycle choice.
    attempts = 0
    while _elapsed() < deadline:
        pieces = _decompose_once(edges, rng, deadline)
        if pieces is None:
            break
        if not _validate(pieces, edges):
            # Should not happen, but guard.
            continue
        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best = pieces
        attempts += 1
        # If we already have a near-optimal bound, we could stop, but keep going
        # within budget to try to improve.
    if best is None:
        # Fallback: all edges as pieces (always valid).
        best = [list(e) for e in edges]
    return best


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # Reserve a small safety margin at the end.
    safety = 15.0
    for idx, spec in enumerate(specs):
        edges = _instance(*spec)
        rng = random.Random(1000 + idx)
        # Allocate remaining budget, leaving safety margin.
        budget_end = _start_time + DEADLINE - safety
        decomp = _decompose_graph(edges, budget_end, rng)
        results.append(decomp)
    return results


if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(i, len(r))
