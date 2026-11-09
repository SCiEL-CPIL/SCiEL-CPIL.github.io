"""Construct a small k-regular graph of girth >= g using random construction + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    rng = random.Random(12345)

    # Moore bound lower estimate (just for reference / sanity)
    # For k-regular girth g:
    #   if g even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    #   if g odd:  n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    if g % 2 == 0:
        mb = 2 * sum((k-1)**i for i in range(g//2))
    else:
        mb = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))

    # Start with a reasonable n: a bit above Moore bound
    # Use known constructions as heuristic: start near 2*k*(k-1)^(g/2-1) for even, etc.
    # For (5,9): Moore bound ~ 1+5*(1+4+16+64)=1+5*85=426. Known best 1116.
    # Try a few candidate sizes.
    
    best_n = None
    best_edges = None

    # Candidate n values to try (from smaller to larger)
    # For k=5, g=9: try around 500-1200
    # General heuristic: n ≈ k*(k-1)^{floor((g-1)/2)} * some_factor
    base = k * (k-1) ** (g // 2)
    candidates = []
    for factor in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        n_try = int(base * factor)
        # Ensure n is at least Moore bound and has correct parity constraints for k-regular
        if n_try < mb:
            n_try = mb
        # n*k must be even for a k-regular graph to exist
        if (n_try * k) % 2 != 0:
            n_try += 1
        if n_try not in candidates:
            candidates.append(n_try)
    candidates.sort()

    # Also add a few smaller ones
    for extra in [int(mb*1.1), int(mb*1.3), int(mb*1.5)]:
        if (extra * k) % 2 != 0:
            extra += 1
        if extra not in candidates and extra >= mb:
            candidates.append(extra)
    candidates = sorted(set(candidates))

    # Time budget per candidate
    time_per_candidate = 80.0 / len(candidates) if candidates else 80.0

    for n in candidates:
        if time.time() > deadline - 10:
            break
        if (n * k) % 2 != 0:
            continue
        if n < 2 * k:  # minimum for k-regular
            continue
        
        cand_deadline = time.time() + min(time_per_candidate, deadline - time.time() - 5)
        
        result = _try_random_kregular(n, k, g, rng, cand_deadline)
        if result is not None:
            n_found, edges = result
            if best_n is None or n_found < best_n:
                best_n = n_found
                best_edges = edges

    if best_n is not None:
        return best_n, best_edges

    # Fallback: return the seed construction if we couldn't find anything
    return _fallback_construct(k, g)


def _try_random_kregular(n, k, g, rng, deadline):
    """Try to construct a k-regular graph of girth >= g on n vertices via random generation + local search."""
    # Step 1: Generate a random k-regular graph
    # Use configuration model (pair up 2n stubs randomly)
    while time.time() < deadline:
        adj = _random_kregular(n, k, rng)
        if adj is None:
            continue
        # Check if simple (no loops/multiedges) - configuration model may create them
        # Verify simplicity
        if not _is_simple(adj, n, k):
            continue
        # Check girth
        girth = _girth_bfs(adj, n)
        if girth >= g:
            edges = _adj_to_edges(adj, n)
            return n, edges
        # Otherwise, try to improve via edge swaps
        improved = _local_search(adj, n, k, g, rng, deadline)
        if improved is not None:
            edges = _adj_to_edges(improved, n)
            return n, edges

    return None


def _random_kregular(n, k, rng):
    """Generate a random k-regular graph using configuration model."""
    if n < k or (n * k) % 2 != 0:
        return None
    # Create stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    # Random permutation
    perm = list(range(n * k))
    rng.shuffle(perm)
    # Pair up
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, n * k, 2):
        u = stubs[perm[i]]
        v = stubs[perm[i+1]]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    if not valid:
        return None
    # Check all degrees
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return adj


def _is_simple(adj, n, k):
    """Check that the graph is simple and k-regular."""
    for u in range(n):
        if len(adj[u]) != k:
            return False
        for v in adj[u]:
            if u == v:
                return False
            if u not in adj[v]:
                return False
    return True


def _girth_bfs(adj, n):
    """Compute the girth (length of shortest cycle) using BFS from each vertex.
    Returns infinity if no cycle (tree), but for k-regular with k>=2 there's always a cycle."""
    min_cycle = float('inf')
    for start in range(n):
        # BFS tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
        if min_cycle == 4:  # Can't do better
            break
    return min_cycle if min_cycle < float('inf') else 2 * n  # fallback


def _local_search(adj, n, k, g, rng, deadline):
    """Try to improve girth via 2-switches (edge swaps)."""
    adj = [set(s) for s in adj]
    best_girth = _girth_bfs(adj, n)
    if best_girth >= g:
        return adj
    
    iterations = 0
    while time.time() < deadline - 1:
        iterations += 1
        if iterations % 1000 == 0 and _girth_bfs(adj, n) >= g:
            return adj
        
        # Pick a random edge (u,v) and another edge (x,y) where {u,v} and {x,y} are disjoint
        # and swapping gives (u,x),(v,y) or (u,y),(v,x) which is still simple
        # Try a few random swaps
        for attempt in range(50):
            # Pick random edge
            u = rng.randint(0, n-1)
            v = rng.choice(list(adj[u]))
            # Pick another random edge disjoint from {u,v}
            x = rng.randint(0, n-1)
            if x in (u, v):
                continue
            candidates_y = [y for y in adj[x] if y not in (u, v, x)]
            if not candidates_y:
                continue
            y = rng.choice(candidates_y)
            
            # Try swap: remove (u,v) and (x,y), add (u,x) and (v,y)
            # Check simplicity
            if x in adj[v] or y in adj[u]:
                continue
            # Perform swap
            adj[u].remove(v); adj[v].remove(u)
            adj[x].remove(y); adj[y].remove(x)
            adj[u].add(x); adj[x].add(u)
            adj[v].add(y); adj[y].add(v)
            
            new_girth = _girth_bfs(adj, n)
            if new_girth >= g:
                return adj
            elif new_girth < best_girth:
                # Revert
                adj[u].remove(x); adj[x].remove(u)
                adj[v].remove(y); adj[y].remove(v)
                adj[u].add(v); adj[v].add(u)
                adj[x].add(y); adj[y].add(x)
            else:
                # Accept if same or better, keep track
                best_girth = new_girth
                # Keep the swap
        # Periodic check
        if iterations % 500 == 0:
            if _girth_bfs(adj, n) >= g:
                return adj

    if _girth_bfs(adj, n) >= g:
        return adj
    return None


def _adj_to_edges(adj, n):
    """Convert adjacency list to edge list (each edge once)."""
    edges = []
    seen = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _fallback_construct(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
