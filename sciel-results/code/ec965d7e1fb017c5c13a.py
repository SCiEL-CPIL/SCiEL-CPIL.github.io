# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(list)
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_long_cycle(adj, edgeset, start, deadline):
    """DFS to find a long simple cycle starting and ending at start."""
    best_cycle = None
    best_len = 0
    # Iterative DFS with path tracking
    # We do a randomized DFS: try neighbors in a shuffled order
    stack = [(start, [start])]
    visited_in_path = {start}
    # Use a recursive-style approach with backtracking
    path = [start]
    path_set = {start}
    # We'll do a randomized DFS with backtracking
    def dfs(u, path, path_set):
        nonlocal best_cycle, best_len
        for v in adj[u]:
            if (min(u, v), max(u, v)) not in edgeset:
                continue
            if v == start and len(path) >= 3:
                cyc = list(path)
                if len(cyc) > best_len:
                    best_len = len(cyc)
                    best_cycle = cyc
                return True  # found a cycle through start
            if v not in path_set:
                path.append(v)
                path_set.add(v)
                dfs(v, path, path_set)
                path.pop()
                path_set.discard(v)
    dfs(start, path, path_set)
    return best_cycle

def _find_cycle_from(adj, edgeset, start):
    """Find a simple cycle through start using DFS."""
    best_cycle = None
    best_len = 0
    path = [start]
    path_set = {start}
    def dfs(u):
        nonlocal best_cycle, best_len
        for v in adj[u]:
            if (min(u, v), max(u, v)) not in edgeset:
                continue
            if v == start and len(path) >= 3:
                cyc = list(path)
                if len(cyc) > best_len:
                    best_len = len(cyc)
                    best_cycle = cyc
                return
            if v not in path_set:
                path.append(v)
                path_set.add(v)
                dfs(v)
                path.pop()
                path_set.discard(v)
    dfs(start)
    return best_cycle

def _decompose(edges, deadline, rng):
    """Decompose edge set into cycles and edges, minimizing piece count."""
    edgeset = set(tuple(sorted(e)) for e in edges)
    if not edgeset:
        return []
    adj = _build_adj(edgeset)
    pieces = []
    remaining = set(edgeset)
    
    # Keep finding cycles until no more can be found
    while remaining:
        if time.time() > deadline:
            break
        # Try to find a cycle: pick a vertex with highest degree in remaining
        deg = defaultdict(int)
        for (u, v) in remaining:
            deg[u] += 1
            deg[v] += 1
        
        # Sort vertices by degree (highest first), with random tie-breaking
        verts = sorted(deg.keys(), key=lambda x: (-deg[x], rng.random()))
        
        found_cycle = None
        for start in verts:
            if deg[start] < 2:
                continue
            # Check if start is in a cycle
            cyc = _find_cycle_from(adj, remaining, start)
            if cyc is not None and len(cyc) >= 3:
                found_cycle = cyc
                break
        
        if found_cycle is None:
            # No more cycles; break remaining edges as single pieces
            for (u, v) in remaining:
                pieces.append([u, v])
            break
        else:
            # Remove cycle edges from remaining
            cyc_edges = set()
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                cyc_edges.add((min(u, v), max(u, v)))
            remaining -= cyc_edges
            pieces.append(found_cycle)
            # Rebuild adj
            adj = _build_adj(remaining)
    
    # Any leftover edges become single pieces
    for (u, v) in remaining:
        pieces.append([u, v])
    
    return pieces

def _decompose_multi(edges, deadline, rng, n_trials=5):
    """Try multiple randomized orderings, keep best."""
    best = None
    best_count = float('inf')
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        # Shuffle vertex order influence by using a fresh rng
        trial_rng = random.Random(rng.randint(0, 10**9))
        pieces = _decompose(edges, deadline, trial_rng)
        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best = pieces
    if best is None:
        best = [[list(e) for e in edges]]
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill
    rng = random.Random(42)
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for (n, p, seed) in specs:
        edges = _instance(n, p, seed)
        # Allocate remaining time across trials
        remaining_time = deadline - time.time()
        pieces = _decompose_multi(edges, deadline, rng, n_trials=3)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
