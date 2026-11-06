"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque
import math

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _get_odd_vertices(edges):
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return [v for v, d in deg.items() if d % 2 != 0]

def _try_merge_paths(paths, edges_set):
    """
    Try to merge two paths if they share an endpoint and the connecting edge exists.
    Returns (merged_paths, success)
    """
    # Build edge set for quick lookup
    edge_lookup = set()
    for u, v in edges_set:
        edge_lookup.add((min(u, v), max(u, v)))
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1 = paths[i]
            p2 = paths[j]
            if len(p1) < 2 or len(p2) < 2:
                continue
            
            # Try all four combinations of endpoints
            for a, b in [(p1[0], p1[-1]), (p1[-1], p1[0])]:
                for c, d in [(p2[0], p2[-1]), (p2[-1], p2[0])]:
                    # Check if a and c are connected
                    e = (min(a, c), max(a, c))
                    if e in edge_lookup:
                        # Merge: p1 + p2 (reversed if needed)
                        if b == c:
                            merged = p1 + p2[1:]
                        elif a == c:
                            merged = [p1[0]] + p2[::-1][1:]
                        else:
                            continue
                        
                        # Verify it's a simple path
                        if len(set(merged)) == len(merged):
                            new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                            new_paths.append(merged)
                            return new_paths, True
    return paths, False

def _greedy_longest_path(n, remaining_edges, start_node, max_depth=200):
    """
    Greedy longest path: extend one edge at a time, preferring edges that
    keep the current endpoint's remaining degree low (to avoid dead ends).
    """
    if not remaining_edges:
        return []
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if start_node not in rem_adj:
        return []
    
    path = [start_node]
    visited = {start_node}
    
    for _ in range(max_depth):
        current = path[-1]
        if current not in rem_adj:
            break
        
        # Get unvisited neighbors
        neighbors = [nb for nb in rem_adj[current] if nb not in visited]
        if not neighbors:
            break
        
        # Score each neighbor: prefer those with higher remaining degree
        # (more options later), but also consider if it leads to a dead end
        def score_neighbor(nb):
            # Higher remaining degree is better (more options)
            deg = len(rem_adj[nb]) - (1 if nb in visited else 0)
            # Penalize if this would be a leaf in the remaining graph
            penalty = -5 if deg <= 1 else 0
            return deg + penalty
        
        # Pick the best neighbor
        best_nb = max(neighbors, key=score_neighbor)
        path.append(best_nb)
        visited.add(best_nb)
    
    return path

def _beam_search_longest_path(n, remaining_edges, start_node, beam_width=20, max_depth=100):
    """
    Finds a long simple path starting from start_node using beam search.
    Returns the path as a list of vertices.
    """
    if not remaining_edges:
        return []
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if start_node not in rem_adj:
        return []

    # Beam: list of (path_tuple, score)
    # Score is length
    beam = [((start_node,), 1)]
    
    for _ in range(max_depth):
        next_beam = []
        for path_tuple, score in beam:
            current = path_tuple[-1]
            visited = set(path_tuple)
            
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            
            if not neighbors:
                # Dead end, keep this path as a candidate
                next_beam.append((path_tuple, score))
                continue
            
            # Generate extensions
            for nb in neighbors:
                new_path = path_tuple + (nb,)
                # Heuristic: prefer nodes with higher remaining degree?
                # Or just length.
                new_score = len(new_path)
                next_beam.append((new_path, new_score))
                
        if not next_beam:
            break
            
        # Sort by score descending and keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
        # Check if any path can be extended
        can_extend = False
        for path_tuple, _ in beam:
            current = path_tuple[-1]
            visited = set(path_tuple)
            if any(nb not in visited for nb in rem_adj[current]):
                can_extend = True
                break
        if not can_extend:
            break
            
    if not beam:
        return []
        
    # Return the longest path
    best_path = max(beam, key=lambda x: len(x[0]))[0]
    return list(best_path)

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    odd_verts = _get_odd_vertices(edges)
    lower_bound = max(1, len(odd_verts) // 2)
    
    best_paths = None
    best_count = float('inf')
    
    start_time = time.time()
    rng = random.Random(seed + 100)
    
    # Strategy: 
    # 1. Greedily build paths starting from odd vertices.
    # 2. Try to merge paths to reduce count.
    # 3. Use beam search for longer paths in later attempts.
    
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
            
        rem_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        
        # Alternate between greedy and beam search
        use_beam = (attempt % 2 == 1)
        
        while rem_edges:
            # Calculate degrees in rem_edges
            rem_deg = defaultdict(int)
            for u, v in rem_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
                
            # Find odd degree vertices in rem_edges
            current_odd = [v for v, d in rem_deg.items() if d % 2 != 0]
            
            if current_odd:
                # Pick a random odd vertex
                start_node = rng.choice(current_odd)
            else:
                # Pick a random vertex with highest degree
                if rem_deg:
                    max_deg = max(rem_deg.values())
                    candidates = [v for v, d in rem_deg.items() if d == max_deg]
                    start_node = rng.choice(candidates)
                else:
                    break
                    
            # Find path using greedy or beam search
            if use_beam:
                path = _beam_search_longest_path(n, rem_edges, start_node, beam_width=20, max_depth=n)
            else:
                path = _greedy_longest_path(n, rem_edges, start_node, max_depth=n)
            
            if not path or len(path) < 2:
                # Fallback: take a single edge
                if rem_edges:
                    incident = [(u, v) for u, v in rem_edges if u == start_node or v == start_node]
                    if incident:
                        u, v = rng.choice(incident)
                        path = [u, v]
                    else:
                        u, v = rng.choice(list(rem_edges))
                        path = [u, v]
                else:
                    break
            
            paths.append(path)
            
            # Remove edges from path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = (min(u, v), max(u, v))
                rem_edges.discard(e)
        
        # Try to merge paths to reduce count
        for _ in range(10):  # Try multiple rounds of merging
            new_paths, merged = _try_merge_paths(paths, edges)
            if not merged:
                break
            paths = new_paths
        
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = [list(p) for p in paths]
            
        if best_count == lower_bound:
            break
            
    if best_paths is None:
        best_paths = [[u, v] for (u, v) in edges]
        
    return best_paths

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 10
            time_budget = max(1.0, min(40.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
