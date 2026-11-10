import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin before 120s limit
    
    n = 20
    # Use a fixed seed for reproducibility if needed, but randomization helps escape local optima
    # Since the task doesn't specify determinism, we can use a fixed seed for consistent "best" found in trials
    # However, usually these tasks allow randomness. Let's use a fixed seed to ensure the output is stable for the evaluator if it runs once.
    # Actually, random restarts are better. But we only have one call to construct().
    # So we must do a single robust run.
    
    random.seed(42)
    
    # Initialize with a high density graph
    p = 0.8
    adj = [[0]*n for _ in range(n)]
    
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                adj[u][v] = 1
                adj[v][u] = 1
                
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def count_common(u, v):
        c = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                c += 1
        return c

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if count_common(u, v) >= 3:
                    return False
        return True

    # Phase 1: Remove all violations
    # We iterate until no violations remain
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    if count_common(u, v) >= 3:
                        adj[u][v] = 0
                        adj[v][u] = 0
                        changed = True
                        # Early exit from inner loop if we want to be faster, but let's check all to ensure stability
                        # Actually, removing one edge might fix multiple pairs.
        
    # Phase 2: Local Search - Try to add back removed edges or add new edges
    # We maintain a list of "candidate" edges that are currently not in the graph
    # We try to add them if they don't create a K2,3
    
    # To speed up, we can pre-compute common neighbors counts?
    # For n=20, O(n^3) is 8000 ops, which is very fast.
    # Let's just re-evaluate.
    
    # Strategy: Randomly shuffle non-edges and try to add them.
    # Repeat this process multiple times (restarts within the time budget)
    
    best_edges = get_edges(adj)
    best_count = len(best_edges)
    
    # We can also try to remove an edge to allow adding two others? (2-opt style)
    # For now, let's stick to greedy addition which is robust.
    
    # Let's do several restarts with different random seeds/densities to find a better local optimum
    # Since we have ~110 seconds, and one pass is very fast (<0.1s), we can do many restarts.
    
    best_adj = [row[:] for row in adj]
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Vary the initial density
        p = 0.7 + random.random() * 0.3
        
        adj = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u][v] = 1
                    adj[v][u] = 1
                    
        # Phase 1: Remove violations
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        if count_common(u, v) >= 3:
                            adj[u][v] = 0
                            adj[v][u] = 0
                            changed = True
                            
        # Phase 2: Greedy Add
        # Create a list of all non-edges
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.append((u, v))
                    
        random.shuffle(non_edges)
        
        added = True
        while added:
            if time.time() > deadline:
                break
            added = False
            for u, v in non_edges:
                if not adj[u][v]:
                    if count_common(u, v) < 3:
                        adj[u][v] = 1
                        adj[v][u] = 1
                        added = True
                        # We don't remove from non_edges list to keep it simple, 
                        # but we check `if not adj[u][v]` so it's safe.
                        # However, iterating the whole list every time is O(E^2).
                        # For n=20, max edges is ~190. 190^2 * 20 (restarts) is fine.
        
        current_edges = get_edges(adj)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_adj = [row[:] for row in adj]
            
        # Optional: Try to improve by removing one edge to add two?
        # This is complex. Greedy add is usually good for this small size.
        # Let's try a few "shake" steps: remove a random edge, try to add back.
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    return get_edges(best_adj)
