import random
import time

def construct():
    n = 40
    m = 401
    # Start with a random graph
    # Ensure connectivity or just random
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            edges.add(tuple(sorted((u, v))))
    
    # Local search
    best_edges = set(edges)
    best_score = float('inf')
    
    def get_score(edge_set):
        # Calculate max book size
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = 0
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    best_score = get_score(best_edges)
    
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    while time.time() - start_time < timeout:
        # Pick a random edge to remove
        if not best_edges:
            break
        e_remove = random.choice(list(best_edges))
        u1, v1 = e_remove
        
        # Pick a random non-edge to add
        # Sample until we find one
        found = False
        attempts = 0
        while not found and attempts < 100:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 != v2:
                e_add = tuple(sorted((u2, v2)))
                if e_add not in best_edges:
                    found = True
                    break
            attempts += 1
        
        if not found:
            continue
            
        # Try the move
        new_edges = set(best_edges)
        new_edges.remove(e_remove)
        new_edges.add(e_add)
        
        new_score = get_score(new_edges)
        
        if new_score <= best_score:
            best_edges = new_edges
            best_score = new_score
            
    # Convert to list of lists
    return [list(e) for e in best_edges]
