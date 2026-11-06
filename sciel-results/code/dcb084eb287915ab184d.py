"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _remove_edge(adj, remaining, u, v):
    e = (min(u, v), max(u, v))
    remaining.discard(e)
    if v in adj[u]:
        adj[u].remove(v)
    if u in adj[v]:
        adj[v].remove(u)

def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices (top by degree + random)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:8] + vertices[:8]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Greedy longest path from start: always extend to unvisited neighbor
        # with highest degree (keeps options open for closing the cycle)
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            # Randomize among top-3 to add diversity
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        # Take the longest such cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Found the longest cycle starting at i

    return best_cycle

def _find_cycles_2opt(adj, n, seed, deadline=None):
    """
    Try to find a cycle using a 2-opt style improvement.
    Start with a random cycle (if exists) and try to extend it.
    """
    if not adj:
        return []
    
    vertices = [v for v in adj if adj[v]]
    if len(vertices) < 3:
        return []
    
    rng = random.Random(seed)
    
    # Try to find a triangle first
    for v in vertices:
        if deadline is not None and time.time() > deadline:
            break
        for u in adj[v]:
            if u <= v:
                continue
            for w in adj[u]:
                if w <= u:
                    continue
                if w in adj[v]:
                    return [v, u, w]
    
    # Try to build a cycle by extending a path
    best_cycle = []
    for _ in range(20):
        if deadline is not None and time.time() > deadline:
            break
        start = rng.choice(vertices)
        path = [start]
        visited = {start}
        
        for _ in range(50):
            u = path[-1]
            candidates = [v for v in adj[u] if v not in visited]
            if not candidates:
                break
            # Prefer neighbors that can close the cycle
            scored = []
            for v in candidates:
                score = len(adj[v])
                if v in adj[path[0]]:
                    score += 100
                scored.append((score, v))
            scored.sort(reverse=True)
            _, next_v = scored[0]
            path.append(next_v)
            visited.add(next_v)
        
        # Check for cycle
        if len(path) >= 3 and path[0] in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)
        else:
            # Check for any cycle in path
            for i in range(len(path) - 2):
                for j in range(len(path) - 1, i, -1):
                    if path[i] in adj[path[j]]:
                        cycle = path[i:j+1]
                        if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                            best_cycle = list(cycle)
                        break
    
    return best_cycle

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    global_deadline = start_time + time_limit

    def greedy_decompose(remaining_edges, n, seed_offset=0, strategy=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)

        cycle_counter = 0
        while remaining:
            # Check time
            if time.time() > global_deadline:
                break
            
            # Choose cycle finding strategy based on strategy parameter
            if strategy == 0:
                cycle = _find_longest_cycle_greedy(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
            elif strategy == 1:
                cycle = _find_cycles_2opt(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
            else:
                # Mixed: try both, take the longer
                c1 = _find_longest_cycle_greedy(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
                c2 = _find_cycles_2opt(adj, n, seed + seed_offset + cycle_counter + 1000, deadline=global_deadline)
                cycle = c1 if len(c1) >= len(c2) else c2
            
            if cycle:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)

                # Verify all cycle edges exist in remaining
                valid = True
                for e in cycle_edges:
                    if e not in remaining:
                        valid = False
                        break
                
                if valid:
                    # Remove from remaining and adj
                    for e in cycle_edges:
                        u, v = e
                        _remove_edge(adj, remaining, u, v)
                    pieces.append(cycle)
                    cycle_counter += 1
                else:
                    # Cycle invalid, fall through to single edge
                    e = next(iter(remaining))
                    u, v = e
                    _remove_edge(adj, remaining, u, v)
                    pieces.append([u, v])
            else:
                # No cycle found, take a single edge
                # Pick an edge that is part of the most potential cycles
                best_edge = None
                best_score = -1
                for e in remaining:
                    u, v = e
                    # Score: number of common neighbors (potential triangles)
                    common = len(set(adj[u]) & set(adj[v]))
                    # Also consider degree
                    deg_score = len(adj[u]) + len(adj[v])
                    total = common * 10 + deg_score
                    if total > best_score:
                        best_score = total
                        best_edge = e
                
                if best_edge is None:
                    best_edge = next(iter(remaining))
                
                u, v = best_edge
                _remove_edge(adj, remaining, u, v)
                pieces.append([u, v])
                
        return pieces

    # Multi-start search with different strategies
    strategies = [0, 1, 2]
    restarts = 0
    for strategy in strategies:
        for restart in range(30):
            if time.time() > global_deadline:
                break
            
            pieces = greedy_decompose(edges, n, restart * 100 + strategy * 1000, strategy)
            current_count = len(pieces)
            
            if current_count < best_count:
                best_count = current_count
                best_pieces = pieces
            restarts += 1
            
            if time.time() > global_deadline:
                break
    
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    total_limit = 145.0  # Safety margin before 150s kill
    
    for i, (n, p, seed) in enumerate(specs):
        remaining_time = total_limit - (time.time() - start_time)
        if remaining_time <= 5:
            # Not enough time, return all single edges for remaining graphs
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=remaining_time - 2)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
