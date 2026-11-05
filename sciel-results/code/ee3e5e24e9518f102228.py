import time
import random

def construct():
    # Grötzsch graph edges
    # Vertices 0-4: cycle
    # Vertices 5-9: copies
    # Vertex 10: apex
    edges = []
    for i in range(5):
        edges.append([i, (i+1)%5])
    for u, w in [(i, (i+1)%5) for i in range(5)]:
        edges.append([u, 5+w])
        edges.append([w, 5+u])
    for u in range(5):
        edges.append([5+u, 10])
    
    # Add vertices 11-15
    # Initially isolated
    n = 16
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_edges = len(edges)
    best_adj = adj[:]
    
    deadline = time.time() + 110
    
    # Helper to check triangle-freeness when adding edge u-v
    def can_add_edge(u, v):
        # Check if u and v share a common neighbor
        return (adj[u] & adj[v]) == 0
        
    # Helper to check 3-colorability
    def is_3_colorable(adj_local):
        n_local = len(adj_local)
        colors = [-1] * n_local
        # Order by degree
        order = sorted(range(n_local), key=lambda i: bin(adj_local[i]).count('1'), reverse=True)
        
        # Precompute neighbors for each vertex in order?
        # Just use adjacency masks
        
        def backtrack(idx):
            if idx == n_local:
                return True
            v = order[idx]
            used_colors = 0
            # Check neighbors
            mask = adj_local[v]
            # Iterate through set bits in mask
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    used_colors |= (1 << colors[u])
                m -= lsb
                
            for c in range(3):
                if not (used_colors & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    # Local search
    # We will try to add edges
    # Generate list of all possible non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Shuffle non-edges
        random.shuffle(all_pairs)
        
        improved = False
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            # If edge already exists, skip
            if (adj[u] >> v) & 1:
                continue
            
            # Check triangle
            if not can_add_edge(u, v):
                continue
            
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check 3-colorability
            if not is_3_colorable(adj):
                # Keep the edge
                improved = True
                # Update best if better
                current_edges = sum(bin(x).count('1') for x in adj) // 2
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
        if not improved:
            # No more edges could be added in this pass
            # Try a different randomization or break
            # Since we shuffle, if no improvement, we might be stuck
            # But we restart the loop, shuffling again
            # To avoid infinite loop of no improvement, we can break or reset
            # However, if we reset to best_adj, we might loop.
            # Let's just continue, as shuffling might allow different orders?
            # No, if no edge can be added without making it 3-colorable or creating triangle,
            # then the graph is "maximal" in this sense.
            # We should break to try a new random start?
            # But we started from Grötzsch.
            # Maybe we should try random restarts from scratch?
            
            # Let's break the inner loop and restart the outer loop with a new random perturbation?
            # Or just break and return best found.
            break

    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result
