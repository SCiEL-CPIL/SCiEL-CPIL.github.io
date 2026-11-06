import time
import random

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140  # 10s margin
    
    # Initialize random graph
    # Ensure we have exactly 91 edges
    # Use a set of edges
    edges = set()
    # Start with a random set of 91 edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    for i in range(m):
        edges.add(all_pairs[i])
        
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def count_c4(adj):
        c4 = 0
        for u in range(n):
            for v in range(u + 1, n):
                s = (adj[u] & adj[v]).bit_count()
                if s > 1:
                    c4 += s * (s - 1) // 2
        return c4 // 2

    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj[:]
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        u, v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        # To speed up, pick random pair and check if it's an edge
        # Or maintain a list of non-edges
        # For n=31, checking is cheap
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edges and (x, y) != (u, v):
                break
        
        # Perform swap
        # Remove (u, v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        edges.discard((u, v))
        
        # Add (x, y)
        adj[x] ^= (1 << y)
        adj[y] ^= (1 << x)
        edges.add((x, y))
        
        new_c4 = count_c4(adj)
        
        # Acceptance criterion
        delta = new_c4 - current_c4
        if delta < 0 or (random.random() < pow(2.71828, -delta / T)):
            current_c4 = new_c4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj[:]
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            edges.add((u, v))
            
            adj[x] ^= (1 << y)
            adj[y] ^= (1 << x)
            edges.discard((x, y))
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    # Reconstruct edges from best_adj
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    return best_edges
