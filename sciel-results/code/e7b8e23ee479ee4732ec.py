import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Initialize Grötzsch Graph (Vertices 0-10)
    # C5: 0-1-2-3-4-0
    # Copies: 5-0, 6-1, 7-2, 8-3, 9-4
    # Center: 10-5, 10-6, 10-7, 10-8, 10-9
    
    def get_initial_adj():
        adj = [set() for _ in range(n)]
        # C5
        for i in range(5):
            u, v = i, (i+1)%5
            adj[u].add(v)
            adj[v].add(u)
        # Copies
        for i in range(5):
            u, v = i, i+5
            adj[u].add(v)
            adj[v].add(u)
        # Center
        for i in range(5):
            u, v = 10, i+5
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    best_edges = []
    best_count = 0
    
    # We can run multiple restarts
    while time.time() < deadline:
        adj = get_initial_adj()
        
        # Create a list of all potential edges
        # We can use a priority queue or just a shuffled list
        # Shuffled list is simpler and effective for small n
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        current_count = len(adj[0]) # Not correct, sum of degrees / 2
        current_count = sum(len(s) for s in adj) // 2
        
        # Greedy addition
        # To speed up, we don't need to rebuild candidates list from scratch,
        # just check validity when we encounter it.
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Check if u and v are still non-adjacent and triangle-free compatible
            if v in adj[u]:
                continue
            
            # Check triangle-free: N(u) and N(v) must be disjoint
            # Note: u is not in N(u), v is not in N(v).
            # We just need intersection of adj[u] and adj[v] to be empty.
            if adj[u] & adj[v]:
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = adj_to_edges(adj)
            
        # Optional: If we reached the theoretical max for this structure, break?
        # Hard to know. Just keep restarting to see if we get lucky with ordering.
        
    return best_edges
