"""Model-and-optimize edge path cover for three fixed graphs."""
import time
import random
import sys
from collections import defaultdict

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="high_deg"):
    """
    Perform a single greedy walk from start_v, returning the path and updating
    remaining and deg in place.
    mode: "high_deg" prefer neighbors with high remaining degree (extend walk),
          "low_deg" prefer neighbors with low remaining degree (classic),
          "balanced" mix of both.
    """
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        # Find valid neighbors
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        if randomize and rng is not None:
            if mode == "high_deg":
                # Prefer high degree neighbors to extend the walk
                weights = [float(deg.get(u, 0) + 1) for u in valid_neighbors]
            elif mode == "low_deg":
                weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
            else:  # balanced
                weights = [1.0 + deg.get(u, 0) * 0.5 for u in valid_neighbors]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            next_v = valid_neighbors[-1]
            for i, u in enumerate(valid_neighbors):
                cum += weights[i]
                if r <= cum:
                    next_v = u
                    break
        else:
            if mode == "high_deg":
                # Deterministic: pick neighbor with highest remaining degree
                next_v = max(valid_neighbors, key=lambda u: deg.get(u, 0))
            elif mode == "low_deg":
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                # Balanced: prefer high degree but break ties by low
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _path_cover_single(edges, deadline, rng=None, randomize=False, mode="high_deg", start_strategy="leaf"):
    """
    Run a single pass of the greedy path cover algorithm.
    Returns list of paths.
    start_strategy: "leaf" (prefer degree-1), "min_deg", "max_deg", "random"
    """
    adj = _build_graph(edges)
    remaining = set(map(tuple, edges))
    paths = []
    
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    while remaining and time.time() < deadline:
        # Find start vertex
        start_candidates = set()
        for e in remaining:
            start_candidates.add(e[0])
            start_candidates.add(e[1])
        
        if start_strategy == "leaf":
            leaves = [v for v in start_candidates if deg[v] == 1]
            if leaves:
                if randomize and rng is not None:
                    start_v = rng.choice(leaves)
                else:
                    start_v = leaves[0]
            else:
                min_deg = min(deg[v] for v in start_candidates)
                candidates = [v for v in start_candidates if deg[v] == min_deg]
                if randomize and rng is not None:
                    start_v = rng.choice(candidates)
                else:
                    start_v = min(candidates)
        elif start_strategy == "max_deg":
            max_deg = max(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == max_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        elif start_strategy == "min_deg":
            min_deg = min(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == min_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        else:  # random
            if randomize and rng is not None:
                start_v = rng.choice(list(start_candidates))
            else:
                start_v = min(start_candidates)
        
        path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize, mode)
        paths.append(path)
    
    # If we ran out of time, add remaining edges as single-edge paths
    for e in remaining:
        paths.append(list(e))
    
    return paths

def _path_cover_with_restarts(edges, deadline):
    """
    Try multiple random orderings of the greedy process to find a better cover.
    """
    best_paths = None
    best_count = len(edges)
    start = time.time()
    
    # First run: deterministic with high-degree preference
    paths = _path_cover_single(edges, deadline, randomize=False, mode="high_deg", start_strategy="leaf")
    best_paths = paths
    best_count = len(paths)
    
    # Also try max_deg start strategy deterministically
    if time.time() < deadline - 10:
        paths2 = _path_cover_single(edges, deadline, randomize=False, mode="high_deg", start_strategy="max_deg")
        if len(paths2) < best_count:
            best_count = len(paths2)
            best_paths = paths2
    
    if time.time() > deadline - 5:
        return best_paths
    
    # Multiple randomized restarts with varied strategies
    rng = random.Random(42)
    
    modes = ["high_deg", "low_deg", "balanced"]
    start_strats = ["leaf", "min_deg", "max_deg", "random"]
    
    trial = 0
    while time.time() < deadline - 10:
        mode = modes[trial % len(modes)]
        strat = start_strats[trial % len(start_strats)]
        
        paths = _path_cover_single(edges, deadline - 5, rng, randomize=True, mode=mode, start_strategy=strat)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            # If we've hit a good number, we can stop early
            if best_count <= 10:
                break
        
        trial += 1
    
    return best_paths

def construct():
    """
    Returns three lists of paths (vertex lists), one per fixed instance graph.
    Uses a model-and-optimize approach with deadline guarding.
    """
    # 150s wall-clock limit. Use 145s to be safe.
    total_deadline = time.time() + 145
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time proportional to edge count
    all_edges = [_instance(n, p, seed) for n, p, seed in specs]
    total_edges = sum(len(e) for e in all_edges)
    
    for i, (n, p, seed) in enumerate(specs):
        edges = all_edges[i]
        # Proportional time allocation with minimum 15s
        frac = len(edges) / total_edges
        time_alloc = max(15.0, frac * 140.0)
        # Ensure we don't exceed remaining time
        remaining_time = total_deadline - time.time()
        time_alloc = min(time_alloc, remaining_time - 2.0)  # 2s buffer
        if time_alloc < 5.0:
            time_alloc = remaining_time - 1.0
        
        graph_deadline = time.time() + time_alloc
        paths = _path_cover_with_restarts(edges, graph_deadline)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
