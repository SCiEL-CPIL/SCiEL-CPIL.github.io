"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_validity(n, all_edges, max_diameter=4):
    """Check if graph is triangle-free, connected, and has diameter <= max_diameter."""
    adj = _build_adj(n, all_edges)
    
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    # Check connected
    visited = set([0])
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    if len(visited) != n:
        return False
    # Check diameter
    for start in range(n):
        dist = {start: 0}
        queue = [start]
        while queue:
            next_queue = []
            for node in queue:
                for neighbor in adj[node]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        next_queue.append(neighbor)
            queue = next_queue
        if len(dist) != n:
            return False
        if max(dist.values()) > max_diameter:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add((min(u,v), max(u,v)))
    
    even_vertices = [i for i in range(n) if i % 2 == 0]
    odd_vertices = [i for i in range(n) if i % 2 == 1]
    
    # All possible edges between even and odd vertices (bipartite)
    possible_edges = []
    for e in even_vertices:
        for o in odd_vertices:
            if (min(e,o), max(e,o)) not in base_edge_set:
                possible_edges.append((e, o))
    
    # Initial solution: hub at 12 connected to all non-adjacent odd vertices
    initial_added = []
    for o in odd_vertices:
        if (min(12,o), max(12,o)) not in base_edge_set:
            initial_added.append([12, o])
    
    best_solution = initial_added
    best_count = len(initial_added)
    
    # Try to improve by local search: remove edges and see if still valid
    # First, try removing edges one at a time
    current = list(best_solution)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            trial = current[:i] + current[i+1:]
            all_edges = base_edges + trial
            if _check_validity(n, all_edges):
                current = trial
                best_count = len(current)
                improved = True
                break
    
    best_solution = current
    best_count = len(current)
    
    # Try adding edges to different hubs and see if we can do better
    # Try each even vertex as a potential hub
    for hub in even_vertices:
        if time.time() > deadline:
            break
        # Build a solution with this hub connected to all odd vertices it's not adjacent to
        added = []
        for o in odd_vertices:
            if (min(hub,o), max(hub,o)) not in base_edge_set:
                added.append([hub, o])
        
        # Try to prune this solution
        trial = list(added)
        improved = True
        while improved:
            improved = False
            for i in range(len(trial)):
                t = trial[:i] + trial[i+1:]
                all_edges = base_edges + t
                if _check_validity(n, all_edges):
                    trial = t
                    improved = True
                    break
        
        if len(trial) < best_count:
            best_solution = trial
            best_count = len(trial)
    
    # Try odd hubs too
    for hub in odd_vertices:
        if time.time() > deadline:
            break
        added = []
        for e in even_vertices:
            if (min(hub,e), max(hub,e)) not in base_edge_set:
                added.append([hub, e])
        
        trial = list(added)
        improved = True
        while improved:
            improved = False
            for i in range(len(trial)):
                t = trial[:i] + trial[i+1:]
                all_edges = base_edges + t
                if _check_validity(n, all_edges):
                    trial = t
                    improved = True
                    break
        
        if len(trial) < best_count:
            best_solution = trial
            best_count = len(trial)
    
    # Try two-hub solutions: one even hub and one odd hub
    # For each pair (even_hub, odd_hub), connect even_hub to some odds and odd_hub to some evens
    # This is complex, so let's try a simpler approach:
    # Start with best single-hub solution and try to add edges from other vertices
    
    # Try greedy: start empty, add edges that most reduce diameter violations
    # But first, let's try a random restart approach
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random subset of possible edges, then prune
        k = random.randint(3, 8)
        random_edges = random.sample(possible_edges, k)
        trial = [[u, v] for u, v in random_edges]
        
        # Prune
        improved = True
        while improved:
            improved = False
            for i in range(len(trial)):
                t = trial[:i] + trial[i+1:]
                all_edges = base_edges + t
                if _check_validity(n, all_edges):
                    trial = t
                    improved = True
                    break
        
        if len(trial) < best_count:
            best_solution = trial
            best_count = len(trial)
    
    # Return as list of [u, v] with Python ints
    result = []
    for u, v in best_solution:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
