"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 140.0  # Safety margin from 150s limit

def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(adj, deg, n, time_limit, rng):
    """
    Find a long cycle in the graph using randomized DFS.
    Returns the cycle as a list of vertices or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try a few random starts and strategies
    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None
    
    # Limit the number of restarts to keep it fast
    max_restarts = 10
    start_time = time.time()
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        start = rng.choice(candidates)
        # DFS with backtracking to find a cycle
        # We use a stack to simulate recursion
        # State: (current_node, path, visited_set)
        # To avoid exponential blowup, we limit path length or use greedy selection
        
        # Greedy DFS: always pick the neighbor that leads to the longest potential path
        # Or just standard DFS with a twist: prioritize unvisited nodes
        
        stack = [(start, [start], {start})]
        found_cycle = None
        
        # Simple DFS for a cycle
        # We want to find ANY cycle, but we want it to be LONG.
        # A simple DFS might find a short one.
        # Let's try to find a cycle that visits many nodes.
        
        # Alternative: Use a heuristic to find a long path first, then check if it closes.
        # For now, let's just do multiple random DFS runs and keep the longest found.
        
        while stack:
            node, path, visited = stack.pop()
            
            # Check if we can close the cycle
            if node == start and len(path) >= 3:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = path[:]
                # Don't break immediately, we might find a longer one? 
                # No, this is DFS, it finds one path. 
                # To find longer ones, we need backtracking.
                # This simple stack DFS is actually just finding *a* cycle, not necessarily the longest.
                # To find a longer one, we need to explore more.
                # Given time constraints, let's just find *a* cycle efficiently.
                # The "Longest" claim in the strategy might be weak if the implementation is just "random DFS".
                # Let's improve: Prioritize neighbors with higher degree to encourage longer paths.
                pass

            neighbors = list(adj[node])
            # Sort neighbors to prefer unvisited or high degree
            neighbors.sort(key=lambda x: (x in visited, -deg[x]))
            
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    # Don't add to stack, this closes the loop
                elif nb not in visited:
                    new_visited = visited | {nb}
                    stack.append((nb, path + [nb], new_visited))
                    
            # Optimization: If we've explored all neighbors and found no cycle, 
            # this branch is dead.
            
            # Break early if we've found a "good enough" cycle? 
            # No, let's run all restarts.
            
        # Reset for next restart
        # Note: The above DFS is not backtracking properly for *longest* path, 
        # it's just a search for *a* cycle. 
        # To get longer cycles, we can try to remove edges that are in short cycles first?
        # No, we remove the cycle we found.
        
    return best_cycle

def _decompose_graph_longest(edges, n, time_limit):
    """
    Decompose graph by repeatedly removing the longest cycle found via randomized DFS.
    """
    if not edges:
        return []

    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    pieces = []
    start_time = time.time()
    rng = random.Random(42) # Fixed seed for reproducibility within the function? Or use global?
    
    while True:
        if time.time() - start_time > time_limit:
            break

        # Check remaining edges
        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break

        # Find a cycle
        cycle = None
        # Try to find a long cycle
        # We can try a few different heuristics
        # 1. Randomized DFS
        # 2. Greedy path extension
        
        # Let's try a simple heuristic: 
        # Find a vertex with max degree.
        # Do a DFS that prefers unvisited nodes.
        # This tends to find longer cycles than "first found".
        
        candidates = [v for v in range(n) if deg[v] >= 2]
        if not candidates:
            # No cycles, take an edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
            continue
            
        # Sort candidates by degree descending
        candidates.sort(key=lambda v: -deg[v])
        
        # Try the top few candidates
        best_cycle = None
        best_len = 0
        
        # Limit search time per step
        step_start = time.time()
        max_search_time = 0.1 # 100ms per step
        
        for start in candidates[:5]: # Try top 5 high-degree vertices
            if time.time() - step_start > max_search_time:
                break
            
            # DFS from start
            stack = [(start, [start], {start})]
            while stack:
                if time.time() - step_start > max_search_time:
                    break
                
                node, path, visited = stack.pop()
                
                # Check for cycle closure
                if node == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    # Don't break, might find longer? 
                    # Actually, in a simple DFS tree, the first cycle found is not necessarily the longest.
                    # But breaking here saves time. Let's keep searching a bit.
                    
                neighbors = list(adj[node])
                # Prioritize neighbors that are unvisited and have high degree
                # This encourages "long" paths
                neighbors.sort(key=lambda x: (x in visited, -deg[x]))
                
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    elif nb not in visited:
                        new_visited = visited | {nb}
                        stack.append((nb, path + [nb], new_visited))
                        
        if best_cycle:
            pieces.append(best_cycle)
            # Remove cycle edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No cycle found in time/effort, take an edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1

    # Add remaining edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    return pieces

def _merge_edges_into_cycles(pieces, edges_set, n, time_limit):
    """
    Post-processing: Try to merge single-edge pieces into cycles.
    Specifically, look for triangles formed by 3 single edges.
    Also, look for 4-cycles formed by 4 single edges.
    """
    start_time = time.time()
    
    # Identify single edge pieces
    edge_pieces = {} # (u,v) -> index
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            edge_pieces[(min(u, v), max(u, v))] = i

    # If no edge pieces, return
    if not edge_pieces:
        return pieces

    # Build adjacency of edge pieces
    # Vertices are the original graph vertices
    # We want to find cycles in the graph formed by ONLY the single edges
    
    adj_single = defaultdict(set)
    for (u, v) in edge_pieces.keys():
        adj_single[u].add(v)
        adj_single[v].add(u)
        
    deg_single = [len(adj_single[v]) for v in range(n)]

    used_pieces = set()
    new_cycles = []
    
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        
        # Find cycles in the single-edge graph
        # Start with triangles
        # For each vertex u, for each pair of neighbors v, w
        # If (v, w) is an edge piece, we have a triangle u-v-w-u
        
        for u in range(n):
            if deg_single[u] < 2:
                continue
            
            neighbors = list(adj_single[u])
            for i in range(len(neighbors)):
                v = neighbors[i]
                if (min(u, v), max(u, v)) in used_pieces:
                    continue
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    if (min(u, w), max(u, w)) in used_pieces:
                        continue
                    # Check if (v, w) is an edge piece
                    edge_vw = (min(v, w), max(v, w))
                    if edge_vw in edge_pieces:
                        if edge_vw in used_pieces:
                            continue
                        
                        # Found triangle u, v, w
                        e_uv = (min(u, v), max(u, v))
                        e_uw = (min(u, w), max(u, w))
                        
                        # Verify these are distinct pieces
                        idx_uv = edge_pieces.get(e_uv)
                        idx_uw = edge_pieces.get(e_uw)
                        idx_vw = edge_pieces.get(edge_vw)
                        
                        if idx_uv is not None and idx_uw is not None and idx_vw is not None:
                            if idx_uv not in used_pieces and idx_uw not in used_pieces and idx_vw not in used_pieces:
                                # Merge them
                                used_pieces.add(idx_uv)
                                used_pieces.add(idx_uw)
                                used_pieces.add(idx_vw)
                                new_cycles.append([u, v, w])
                                changed = True
                                # Update local state for this iteration? 
                                # We break out to restart the loop
                                break
                if changed:
                    break
            if changed:
                break
        
        # Also try 4-cycles? Might be too slow/complex. Triangles are most common.

    if not changed:
        return pieces

    # Rebuild pieces
    final_pieces = []
    for i, piece in enumerate(pieces):
        if len(piece) > 2:
            final_pieces.append(piece)
        elif len(piece) == 2:
            u, v = piece
            edge = (min(u, v), max(u, v))
            idx = edge_pieces.get(edge)
            if idx is not None and idx not in used_pieces:
                final_pieces.append(piece)
        
    final_pieces.extend(new_cycles)
    
    return final_pieces

def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 10:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Allocate time for this graph
        # Use 90% of remaining time for the main decomposition
        graph_time = remaining * 0.9

        # Strategy: Greedy Longest Cycle Removal
        pieces = _decompose_graph_longest(edges, n, graph_time * 0.8)
        
        # Post-processing: Merge edge pieces into cycles
        pieces = _merge_edges_into_cycles(pieces, edges_set, n, graph_time * 0.2)

        results.append(pieces)

    return results
