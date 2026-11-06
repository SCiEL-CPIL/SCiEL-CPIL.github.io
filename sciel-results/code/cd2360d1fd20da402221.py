import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Construct Grötzsch Graph on vertices 0..10
    # Cycle 0-1-2-3-4-0
    # Copies 5-9 where i+5 connected to N(i)
    # Apex 10 connected to 5-9
    edges = []
    # Cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Mycielski edges
    for i in range(5):
        # i+5 connected to neighbors of i in cycle
        n1 = (i - 1) % 5
        n2 = (i + 1) % 5
        edges.append((5 + i, n1))
        edges.append((5 + i, n2))
    # Apex
    for i in range(5):
        edges.append((10, 5 + i))
    
    n_vertices = 16
    # Convert to set of frozensets for fast lookup
    edge_set = set(frozenset(e) for e in edges)
    
    # Adjacency list for fast triangle check
    adj = [set() for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def is_bipartite():
        colors = [-1] * n_vertices
        for start in range(n_vertices):
            if colors[start] != -1:
                continue
            colors[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        return False
        return True

    def is_3_colorable():
        if is_bipartite():
            return True
        # Backtracking 3-coloring
        colors = [-1] * n_vertices
        
        def backtrack(v):
            if v == n_vertices:
                return True
            # Heuristic: try colors 0, 1, 2
            for c in range(3):
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Initial check: Grötzsch is not 3-colorable and triangle-free
    # assert not is_3_colorable()
    
    best_edges = set(edge_set)
    best_count = len(best_edges)
    
    # Precompute all potential edges
    all_possible_edges = set()
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            all_possible_edges.add(frozenset((u, v)))
            
    # List of edges not in the graph
    missing_edges = list(all_possible_edges - edge_set)
    
    random.seed(42)
    
    def can_add_edge(e):
        u, v = tuple(e)
        # Check triangle
        if adj[u] & adj[v]:
            return False
        # Tentative add
        adj[u].add(v)
        adj[v].add(u)
        
        colorable = is_3_colorable()
        
        # Revert
        adj[u].remove(v)
        adj[v].remove(u)
        
        return not colorable

    def can_remove_edge(e):
        u, v = tuple(e)
        # Tentative remove
        adj[u].remove(v)
        adj[v].remove(u)
        
        colorable = is_3_colorable()
        
        # Restore
        adj[u].add(v)
        adj[v].add(u)
        
        return not colorable

    iterations = 0
    current_edges = set(edge_set)
    current_count = len(current_edges)
    
    # Local Search Loop
    while time.time() < deadline:
        iterations += 1
        
        # Randomly pick a move
        if random.random() < 0.8: # Bias towards adding
            if not missing_edges:
                break
            e = random.choice(missing_edges)
            if e in current_edges:
                continue
            if can_add_edge(e):
                current_edges.add(e)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
        else:
            # Remove an edge
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            if can_remove_edge(e):
                current_edges.remove(e)
                current_count -= 1
                
    # Convert best_edges to list of lists
    final_edges = [list(e) for e in best_edges]
    return final_edges
