import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # Initialize with 5 disjoint K4s
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                u = base + a
                v = base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    def try_add_edge(u, v):
        if (adj[u] >> v) & 1:
            return True # Already exists
        
        # Check 1: u, v common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
            
        # Check 2: u, x common neighbors where v is in N(x)
        c = adj[u]
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c -= lsb
            if x == v:
                continue
            if (adj[x] >> v) & 1:
                if (adj[u] & adj[x]).bit_count() >= 2:
                    return False
        
        # Check 3: v, x common neighbors where u is in N(x)
        c = adj[v]
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c -= lsb
            if x == u:
                continue
            if (adj[x] >> u) & 1:
                if (adj[v] & adj[x]).bit_count() >= 2:
                    return False
        
        return True

    def get_edge_count():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    best_edges = get_edge_count()
    best_adj = adj[:]
    
    # Local Search
    # We will run multiple restarts or just one long hill climb with random restarts if stuck
    # Given the small size, a single long run with "remove random edge" when stuck is good.
    
    while time.time() < deadline:
        added_any = False
        # Generate list of missing edges
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not ((adj[i] >> j) & 1)]
        random.shuffle(missing)
        
        for u, v in missing:
            if try_add_edge(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added_any = True
                
        if not added_any:
            # Stuck. Try to remove a random edge to break symmetry or bottleneck
            if get_edge_count() > 0:
                existing = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
                if existing:
                    u, v = random.choice(existing)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                break
                
        current_edges = get_edge_count()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
    # Return the best found
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
