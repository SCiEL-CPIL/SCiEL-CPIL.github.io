import random
import time

def construct():
    n = 20
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, u, v):
        # Check if adding edge (u, v) creates a K2,3
        # This happens if u and v have >= 3 common neighbors
        common = adj[u] & adj[v]
        return popcount(common) < 3

    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    best_edges = 0
    best_adj = [0] * n
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # We will use a local search strategy with random restarts
    num_restarts = 2000 # High number, but controlled by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start with empty graph
        curr_adj = [0] * n
        
        # Greedy phase: add edges in random order
        random.shuffle(pairs)
        for u, v in pairs:
            if not (curr_adj[u] >> v) & 1: # if not already added
                if can_add(curr_adj, u, v):
                    curr_adj[u] |= (1 << v)
                    curr_adj[v] |= (1 << u)
        
        curr_count = count_edges(curr_adj)
        
        # Local improvement phase:
        # Try removing each edge and see if we can gain more
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get list of current edges
            edges_to_try = []
            for i in range(n):
                mask = curr_adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges_to_try.append((i, j))
                    mask ^= lsb
            
            if not edges_to_try:
                break
                
            random.shuffle(edges_to_try)
            
            # Try a subset of edges to save time if needed, or all
            # For n=20, max edges ~40-50, so iterating all is fine
            for u, v in edges_to_try:
                if time.time() > deadline:
                    break
                    
                # Snapshot current state
                old_adj = [x for x in curr_adj]
                
                # Remove (u, v)
                curr_adj[u] &= ~(1 << v)
                curr_adj[v] &= ~(1 << u)
                
                # Re-saturate: try to add any non-existent edge
                # We use a greedy approach again
                random.shuffle(pairs)
                for a, b in pairs:
                    if not (curr_adj[a] >> b) & 1:
                        if can_add(curr_adj, a, b):
                            curr_adj[a] |= (1 << b)
                            curr_adj[b] |= (1 << a)
                            
                new_count = count_edges(curr_adj)
                
                if new_count > curr_count:
                    curr_count = new_count
                    improved = True
                else:
                    # Revert
                    curr_adj = old_adj
                    
        if curr_count > best_edges:
            best_edges = curr_count
            best_adj = [x for x in curr_adj]
            
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
