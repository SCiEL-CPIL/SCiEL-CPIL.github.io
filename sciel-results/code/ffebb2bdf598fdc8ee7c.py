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

def _max_2_matching_greedy(edges, deadline, rng=None):
    """
    Find a large 2-matching: subset of edges where each vertex has degree <= 2.
    Uses greedy initialization + local search refinement.
    Returns the set of edges in the 2-matching.
    """
    adj = _build_graph(edges)
    all_edges = set(map(tuple, edges))
    
    # Greedy initialization: process edges in order, add if doesn't violate degree constraint
    deg = defaultdict(int)
    matching = set()
    
    # Try multiple edge orderings
    best_matching = set()
    best_size = 0
    
    orderings = []
    # Original order
    orderings.append(list(all_edges))
    # Random orderings
    if rng is not None:
        for _ in range(20):
            shuffled = list(all_edges)
            rng.shuffle(shuffled)
            orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        deg = defaultdict(int)
        matching = set()
        
        for e in ordering:
            u, v = e
            if deg[u] < 2 and deg[v] < 2:
                matching.add(e)
                deg[u] += 1
                deg[v] += 1
        
        if len(matching) > best_size:
            best_size = len(matching)
            best_matching = matching.copy()
    
    # Local search: try to add back edges or swap to improve
    # An edge can be added if both endpoints have deg < 2
    # We can also try: remove an edge incident to u, add two edges to increase matching
    # For simplicity, try adding any edge that fits
    deg = defaultdict(int)
    for e in best_matching:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in all_edges:
            if e in best_matching:
                continue
            u, v = e
            if deg[u] < 2 and deg[v] < 2:
                best_matching.add(e)
                deg[u] += 1
                deg[v] += 1
                improved = True
    
    # Try edge swaps: for each edge e1 in matching, try removing it and adding up to 2 new edges
    # This is complex; skip for now and rely on the greedy + add-back
    
    return best_matching

def _extract_paths_from_2matching(matching_edges, n):
    """
    Given a 2-matching (edges where each vertex has deg <= 2), extract paths and cycles.
    Returns list of paths (vertex lists). Cycles are broken into paths by removing one edge.
    """
    if not matching_edges:
        return []
    
    adj = defaultdict(list)
    for u, v in matching_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find connected components
    visited = set()
    components = []
    
    for v in range(n):
        if v in visited or v not in adj:
            continue
        # BFS/DFS to find component
        comp = []
        stack = [v]
        visited.add(v)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(comp)
    
    paths = []
    for comp in components:
        # Find vertices with degree 1 (path endpoints) in this component
        deg_in_comp = defaultdict(int)
        for u, v in matching_edges:
            if u in comp and v in comp:
                deg_in_comp[u] += 1
                deg_in_comp[v] += 1
        
        endpoints = [v for v in comp if deg_in_comp.get(v, 0) == 1]
        
        if len(endpoints) == 2:
            # It's a path
            start = endpoints[0]
            path = [start]
            visited_in_path = {start}
            while len(path) < len(comp):
                v = path[-1]
                next_v = None
                for nb in adj[v]:
                    if nb not in visited_in_path:
                        next_v = nb
                        break
                if next_v is None:
                    break
                path.append(next_v)
                visited_in_path.add(next_v)
            paths.append(path)
        elif len(endpoints) == 0:
            # It's a cycle - break it into a path by removing one edge
            # Find a vertex with degree 2, start from it
            if comp:
                start = comp[0]
                path = [start]
                visited_in_path = {start}
                # Walk around the cycle
                while len(path) < len(comp):
                    v = path[-1]
                    next_v = None
                    for nb in adj[v]:
                        if nb not in visited_in_path:
                            next_v = nb
                            break
                    if next_v is None:
                        break
                    path.append(next_v)
                    visited_in_path.add(next_v)
                # path now contains all vertices of the cycle (in order)
                # To make it a simple path (not cycle), just use it as is
                # The last vertex connects back to first, but we don't include that edge
                paths.append(path)
        else:
            # Shouldn't happen in a 2-matching, but handle gracefully
            for v in comp:
                paths.append([v])
    
    return paths

def _cover_remaining_edges(paths, all_edges, deadline, rng=None):
    """
    Find edges not covered by the 2-matching paths, and add them as additional paths.
    """
    covered_edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            covered_edges.add((min(u, v), max(u, v)))
    
    remaining = set(map(tuple, all_edges)) - covered_edges
    
    if not remaining:
        return paths
    
    # Build graph of remaining edges
    adj = _build_graph(list(remaining))
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    # Greedily add paths for remaining edges
    paths_copy = list(paths)
    remaining_set = set(remaining)
    
    while remaining_set and time.time() < deadline:
        # Find a start vertex
        start_candidates = set()
        for e in remaining_set:
            start_candidates.add(e[0])
            start_candidates.add(e[1])
        
        if not start_candidates:
            break
        
        # Prefer leaves
        leaves = [v for v in start_candidates if deg[v] == 1]
        if leaves:
            if rng:
                start_v = rng.choice(leaves)
            else:
                start_v = leaves[0]
        else:
            min_deg = min(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == min_deg]
            if rng:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        
        # Greedy walk
        path = [start_v]
        visited = {start_v}
        while True:
            v = path[-1]
            valid_neighbors = []
            for u in adj.get(v, []):
                if u not in visited:
                    e = (min(v, u), max(v, u))
                    if e in remaining_set:
                        valid_neighbors.append(u)
            
            if not valid_neighbors:
                break
            
            # Choose neighbor with lowest degree
            next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            e = (min(v, next_v), max(v, next_v))
            remaining_set.discard(e)
            deg[v] -= 1
            deg[next_v] -= 1
            path.append(next_v)
            visited.add(next_v)
        
        if len(path) >= 2:
            paths_copy.append(path)
        else:
            # Single vertex, shouldn't happen if we have edges
            break
    
    # Add any truly remaining edges as single-edge paths
    for e in remaining_set:
        paths_copy.append(list(e))
    
    return paths_copy

def _try_merge_paths(paths, all_edges, deadline, rng=None):
    """
    Try to merge two paths if they share an endpoint and there's an unused edge connecting them.
    Actually, since paths are edge-disjoint and cover all edges, we can't merge without violating constraints.
    Instead, try to restructure: if two paths have endpoints that are connected by an edge in the original graph
    that is currently used by one of the paths, we might be able to rearrange.
    For now, this is a placeholder for future improvement.
    """
    return paths

def _path_cover_2matching(edges, deadline, rng=None):
    """
    Main algorithm: use maximum 2-matching to minimize path count.
    """
    n = max(max(u, v) for u, v in edges) + 1 if edges else 0
    
    # Find max 2-matching
    matching = _max_2_matching_greedy(edges, deadline - 5, rng)
    
    # Extract paths from 2-matching
    paths = _extract_paths_from_2matching(matching, n)
    
    # Cover remaining edges
    paths = _cover_remaining_edges(paths, edges, deadline - 2, rng)
    
    return paths

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="low_deg"):
    """
    Greedy walk that builds a simple path by extending from start_v.
    """
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
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
            else:
                weights = [1.0 + deg.get(u, 0) * 0.5 for u in valid_neighbors]
            
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
                next_v = max(valid_neighbors, key=lambda u: deg.get(u, 0))
            elif mode == "low_deg":
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _path_cover_single(edges, deadline, rng=None, randomize=False, mode="low_deg", start_strategy="leaf", shuffle_adj=False):
    """
    Single pass greedy path cover.
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
        else:
            if randomize and rng is not None:
                start_v = rng.choice(list(start_candidates))
            else:
                start_v = min(start_candidates)
        
        path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize, mode)
        paths.append(path)
    
    for e in remaining:
        paths.append(list(e))
        
    return paths

def _path_cover_with_restarts(edges, deadline):
    """
    Try multiple strategies: 2-matching approach + greedy restarts.
    """
    best_paths = None
    best_count = len(edges)
    start = time.time()
    
    # 1. 2-matching approach
    if time.time() < deadline - 10:
        rng = random.Random(123)
        paths_2m = _path_cover_2matching(edges, deadline - 5, rng)
        if paths_2m:
            best_paths = paths_2m
            best_count = len(paths_2m)
    
    # 2. Deterministic baseline: leaf-start, low-degree greedy
    paths_g = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="leaf")
    if len(paths_g) < best_count:
        best_count = len(paths_g)
        best_paths = paths_g
    
    # 3. More randomized restarts
    rng = random.Random(42)
    modes = ["low_deg", "low_deg", "balanced", "high_deg"]
    start_strats = ["leaf", "min_deg", "random"]
    
    trial = 0
    while time.time() < deadline - 2:
        mode = modes[trial % len(modes)]
        strat = start_strats[trial % len(start_strats)]
        
        paths = _path_cover_single(edges, deadline - 2, rng, randomize=True, mode=mode, start_strategy=strat, shuffle_adj=True)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            if best_count <= 8:
                break
        
        trial += 1
    
    return best_paths

def construct():
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
