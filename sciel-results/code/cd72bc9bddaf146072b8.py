import random
import time

def construct():
    n = 20
    start_time = time.time()
    limit = 110.0 # Safety margin
    
    def popcount(x):
        return bin(x).count("1")
        
    best_edges = []
    best_count = 0
    
    def is_valid_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # Check 1: Common neighbors of u, v
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # Check 2: For w in N(v), check |N(u) & N(w)| <= 1
        # N(v) is adj[v]
        # We need to iterate over bits in adj[v]
        mask_v = adj[v]
        # Exclude u if it was in N(v) (it isn't, since not connected)
        # Iterate set bits in mask_v
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w == u: # Should not happen if not connected
                m = m & (m - 1)
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False
            m = m & (m - 1)
            
        # Check 3: For w in N(u), check |N(v) & N(w)| <= 1
        mask_u = adj[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w == v:
                m = m & (m - 1)
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
            m = m & (m - 1)
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges(adj):
        return sum(bin(adj[u]).count("1") for u in range(n)) // 2

    # Local search
    def local_search(adj, max_iter=1000):
        improved = True
        while improved and (time.time() - start_time) < limit:
            improved = False
            # Try to add random edge
            random.shuffle(list(range(n)))
            for u in range(n):
                for v in range(u + 1, n):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        improved = True
                        # Continue trying to add more
            # If no improvement, kick
            if not improved:
                # Remove 2-3 random edges
                edges = get_edges(adj)
                if edges:
                    for _ in range(2):
                        if edges:
                            e = random.choice(edges)
                            remove_edge(adj, e[0], e[1])
                            edges.remove(e)
        
    # Run multiple restarts
    while time.time() - start_time < limit:
        adj = [0] * n
        # Random initialization: add random edges
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.3:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
        
        local_search(adj)
        
        curr_count = count_edges(adj)
        if curr_count > best_count:
            best_count = curr_count
            best_edges = get_edges(adj)
            
    return best_edges

construct()
