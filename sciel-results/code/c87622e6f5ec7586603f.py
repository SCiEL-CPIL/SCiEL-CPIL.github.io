import random
import time

def construct():
    n = 40
    target_edges = 401
    
    # Initialize random graph
    # Start with a random set of edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    edges = set(all_pairs[:target_edges])
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    def get_max_book():
        max_b = 0
        for u, v in edges:
            # Count common neighbors
            common = (adj[u] & adj[v]).bit_count()
            if common > max_b:
                max_b = common
        return max_b
        
    current_score = get_max_book()
    best_score = current_score
    best_edges = set(edges)
    best_adj = adj[:]
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    T = 10.0
    cooling = 0.999
    
    # Precompute list of all pairs for random selection
    # We need to pick a random edge and a random non-edge.
    # Iterating over all_pairs to find non-edge is O(n^2).
    # Better: maintain a list of non-edges?
    # Or just pick random u, v and check if in edges.
    
    while time.time() < deadline:
        # Pick random edge to remove
        if not edges: break
        u, v = random.choice(list(edges))
        
        # Pick random non-edge to add
        # Try a few times to find a valid non-edge
        x, y = u, v
        while (x, y) in edges or (y, x) in edges or x == y:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x > y: x, y = y, x
            if x == y: continue
            
        # Apply move
        # Remove (u,v)
        edges.remove((u,v))
        adj[u] ^= 1 << v
        adj[v] ^= 1 << u
        
        # Add (x,y)
        edges.add((x,y))
        adj[x] ^= 1 << y
        adj[y] ^= 1 << x
        
        new_score = get_max_book()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edges)
                best_adj = adj[:]
        else:
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta / T):
                current_score = new_score
            else:
                # Revert
                edges.remove((x,y))
                adj[x] ^= 1 << y
                adj[y] ^= 1 << x
                
                edges.add((u,v))
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # Reconstruct best edges
    return [[u, v] for u, v in best_edges]
