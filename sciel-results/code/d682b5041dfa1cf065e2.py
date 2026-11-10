"""
Path cover for three fixed graphs (Gallai's conjecture).
Strategy:
1. Generate near-optimal path covers via randomized greedy walks with multiple strategies.
2. Post-process with iterative path merging (endpoint matching + disjointness check).
3. Use deadline guards to stay within the 150s wall-clock limit.
"""
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

def _greedy_walk(adj, remaining, deg, start_v, rng, randomize, mode):
    """
    Perform a single greedy walk from start_v.
    mode: "low_deg" (prefer neighbors with low remaining degree),
          "high_deg" (prefer neighbors with high remaining degree),
          "balanced" (mix),
          "random" (uniform random).
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
                weights = [float(deg.get(u, 0) + 1) for u in valid_neighbors]
            elif mode == "low_deg":
                weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
            elif mode == "random":
                weights = [1.0] * len(valid_neighbors)
            else:  # balanced
                weights = [1.0 + deg.get(u, 0) * 0.3 for u in valid_neighbors]
            
            total_w = sum(weights)
            if total_w == 0:
                next_v = valid_neighbors[0]
            else:
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
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
            elif mode == "low_deg":
                next_v = min(valid_neighbors, key=lambda u: (deg.get(u, 0), u))
            elif mode == "random":
                next_v = valid_neighbors[0]
            else:  # balanced
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _try_merge_paths(paths, adj):
    """
    Iteratively merge paths that share an endpoint and are otherwise disjoint.
    Returns the merged list of paths.
    """
    if len(paths) < 2:
        return paths
    
    active_paths = [list(p) for p in paths]
    path_sets = [set(p) for p in active_paths]
    
    changed = True
    while changed:
        changed = False
        # Rebuild endpoint map
        v_map = defaultdict(lambda: {"starts": [], "ends": []})
        for i, p in enumerate(active_paths):
            if p:
                v_map[p[0]]["starts"].append(i)
                v_map[p[-1]]["ends"].append(i)
        
        found = False
        for v, entries in v_map.items():
            starts = entries["starts"]
            ends = entries["ends"]
            
            for i in ends:
                for j in starts:
                    if i == j:
                        continue
                    # Check disjointness: intersection should be exactly {v}
                    inter = path_sets[i] & path_sets[j]
                    if inter == {v}:
                        # Merge: path i (ends at v) + path j (starts at v) -> i + j[1:]
                        new_path = active_paths[i] + active_paths[j][1:]
                        new_set = path_sets[i] | path_sets[j]
                        
                        # Remove i and j (careful with indices)
                        if i > j:
                            i, j = j, i
                        active_paths.pop(j)
                        active_paths.pop(i)
                        path_sets.pop(j)
                        path_sets.pop(i)
                        
                        active_paths.append(new_path)
                        path_sets.append(new_set)
                        
                        changed = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    return active_paths

def _path_cover_single(edges, deadline, rng, randomize, mode, start_strategy, shuffle_adj):
    """
    Run a single pass of the greedy path cover algorithm.
    """
    adj = _build_graph(edges)
    
    if shuffle_adj and rng is not None:
        for v in adj:
            rng.shuffle(adj[v])
    
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
        
    # Post-processing: Try to merge paths
    if time.time() < deadline - 1.0:
        paths = _try_merge_paths(paths, adj)
        
    return paths

def _path_cover_with_restarts(edges, deadline):
    """
    Try multiple random orderings of the greedy process to find a better cover.
    """
    best_paths = None
    best_count = len(edges)
    start = time.time()
    
    # Deterministic baselines
    paths = _path_cover_single(edges, deadline, None, False, "low_deg", "leaf", False)
    best_paths = paths
    best_count = len(paths)
    
    if time.time() < deadline - 10:
        paths2 = _path_cover_single(edges, deadline, None, False, "low_deg", "min_deg", False)
        if len(paths2) < best_count:
            best_count = len(paths2)
            best_paths = paths2
    
    if time.time() < deadline - 10:
        paths3 = _path_cover_single(edges, deadline, None, False, "balanced", "leaf", False)
        if len(paths3) < best_count:
            best_count = len(paths3)
            best_paths = paths3

    if time.time() > deadline - 5:
        return best_paths
    
    # Randomized restarts
    rng = random.Random(42)
    
    modes = ["low_deg", "low_deg", "balanced", "high_deg", "random"]
    start_strats = ["leaf", "min_deg", "random", "max_deg"]
    
    trial = 0
    while time.time() < deadline - 5:
        mode = modes[trial % len(modes)]
        strat = start_strats[trial % len(start_strats)]
        
        paths = _path_cover_single(edges, deadline - 5, rng, True, mode, strat, True)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            if best_count <= 10:
                break
        
        trial += 1
    
    return best_paths

def construct():
    """
    Returns three lists of paths (vertex lists), one per fixed instance graph.
    """
    total_deadline = time.time() + 145
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    all_edges = [_instance(n, p, seed) for n, p, seed in specs]
    total_edges = sum(len(e) for e in all_edges)
    
    for i, (n, p, seed) in enumerate(specs):
        edges = all_edges[i]
        frac = len(edges) / total_edges
        time_alloc = max(15.0, frac * 140.0)
        remaining_time = total_deadline - time.time()
        time_alloc = min(time_alloc, remaining_time - 2.0)
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
