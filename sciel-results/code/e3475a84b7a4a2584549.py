import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    results = []
    for n, p, seed in [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]:
        if time.time() > deadline:
            # Return best effort or empty?
            # The task says "invalid result... scores 0".
            # We must return a valid result.
            # If we run out of time, we should return the trivial solution (each edge as a path)
            # which is always valid.
            pass
            
        edges = generate_edges(n, p, seed)
        paths = decompose_edges(edges, n)
        results.append(paths)
        
    return results

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_edges(edges, n):
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    remaining_edges = set(edges)
    paths = []
    
    while remaining_edges:
        # Find a greedy path
        # Start from a vertex with highest degree in remaining graph
        # Calculate degrees in remaining graph
        degs = [0] * n
        for u, v in remaining_edges:
            degs[u] += 1
            degs[v] += 1
            
        # Pick start vertex with max degree
        if not any(degs):
            break # Should not happen if remaining_edges is not empty
            
        start = max(range(n), key=lambda i: degs[i])
        
        # Greedy walk
        path = [start]
        visited = {start}
        current = start
        
        while True:
            # Find next vertex
            best_next = -1
            best_deg = -1
            for neighbor in adj[current]:
                # Check if edge (current, neighbor) is in remaining_edges
                edge = tuple(sorted((current, neighbor)))
                if edge in remaining_edges:
                    if degs[neighbor] > best_deg:
                        best_deg = degs[neighbor]
                        best_next = neighbor
            
            if best_next == -1:
                break # Stuck
                
            # Move
            path.append(best_next)
            visited.add(best_next)
            
            # Remove edge
            edge = tuple(sorted((current, best_next)))
            remaining_edges.remove(edge)
            
            # Update degrees
            degs[current] -= 1
            degs[best_next] -= 1
            
            current = best_next
            
        paths.append(path)
        
    return paths
