import random
import time
from collections import defaultdict, deque

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges, time_limit=30.0):
    """
    Decomposes a graph given by a list of edges into cycles and single edges.
    Returns a list of pieces. A piece is a list of vertices.
    If len(piece) == 2, it's an edge. If len(piece) >= 3, it's a cycle.
    """
    best_pieces = None
    best_count = float('inf')
    start_time = time.time()
    
    # Estimate max restarts based on time, but cap it to prevent runaway
    # A single decomposition pass is roughly O(E * (V+E)) with DFS.
    # For N=30, E~135, this is fast. We can do many restarts.
    # Let's try up to 200 restarts but break on time.
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        current_edges = set(edges)
        pieces = []
        
        while current_edges:
            # Check time inside the loop too, just in case
            if time.time() - start_time > time_limit:
                return best_pieces # Return best found so far if we run out of time mid-restart
            
            # Pick a random edge
            u, v = random.choice(list(current_edges))
            current_edges.remove((u, v))
            
            # Build adjacency list for remaining edges
            adj = defaultdict(set)
            for a, b in current_edges:
                adj[a].add(b)
                adj[b].add(a)
                
            # Find path from u to v
            path = find_path_randomized_iter(u, v, adj)
            
            if path:
                # path is [u, ..., v]
                cycle_vertices = path
                pieces.append(cycle_vertices)
                
                # Remove all edges in the path from current_edges
                for i in range(len(cycle_vertices) - 1):
                    e1 = cycle_vertices[i]
                    e2 = cycle_vertices[i+1]
                    edge_to_remove = tuple(sorted((e1, e2)))
                    if edge_to_remove in current_edges:
                        current_edges.remove(edge_to_remove)
            else:
                # No path found, (u, v) is a bridge or isolated in remaining graph
                pieces.append([u, v])
                
        # Evaluate this decomposition
        count = len(pieces)
        if count < best_count:
            best_count = count
            best_pieces = pieces
            
    return best_pieces

def find_path_randomized_iter(start, end, adj):
    """
    Finds a path from start to end in the graph defined by adj.
    Uses randomized iterative DFS.
    Returns list of nodes [start, ..., end] or None.
    """
    if start == end:
        return [start]
        
    visited = {start}
    stack = [(start, [start])]
    
    while stack:
        node, path = stack.pop()
        if node == end:
            return path
        
        neighbors = list(adj[node])
        random.shuffle(neighbors)
        
        for nei in neighbors:
            if nei not in visited:
                visited.add(nei)
                new_path = path + [nei]
                stack.append((nei, new_path))
                
    return None

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    global_start = time.time()
    global_deadline = 115.0 # 5s safety margin
    
    for spec in specs:
        # Check if we have enough time left for the next graph
        elapsed = time.time() - global_start
        if elapsed > global_deadline:
            # If we run out of time, return empty lists or best effort?
            # The contract says "three lists". If we can't compute one, it's invalid.
            # We must ensure we finish.
            # Let's break and return what we have, padded with empty lists?
            # No, "invalid result ... scores 0". We MUST return valid decompositions.
            # So we must strictly manage time to ensure all 3 finish.
            pass
            
        edges = instance_graph(*spec)
        
        # Allocate time budget per graph.
        # 3 graphs. Total 115s.
        # Let's give 35s to each, but check global time.
        current_time = time.time()
        time_remaining = global_deadline - current_time
        # We have 2 more graphs after this one (if this is the first).
        # Simple strategy: 1/3 of remaining time, but capped.
        # Or just fixed 35s. 35*3 = 105s. Safe.
        alloc_time = min(35.0, time_remaining)
        
        pieces = decompose_graph(edges, time_limit=alloc_time)
        results.append(pieces)
        
        # If we are running out of time, subsequent graphs might fail.
        # But 35s * 3 = 105s is well under 115s.
        
    return results
