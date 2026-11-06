import random
import time

def construct():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute bit indices
    # adj is a list of ints
    
    def can_add(u, v, adj):
        # u, v not in adj
        mask_u = adj[u]
        mask_v = adj[v]
        
        # Check u's neighbors against v's neighbors
        # For each w in adj[v], check if (adj[u] & adj[w]).count >= 2
        w = mask_v
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (mask_u & adj[idx]).bit_count() >= 2:
                return False
        
        # Check v's neighbors against u's neighbors
        w = mask_u
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (mask_v & adj[idx]).bit_count() >= 2:
                return False
        
        return True

    def greedy_construction(order):
        adj = [0] * n
        edges = []
        for u, v in order:
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return edges, adj

    def local_search(edges, adj):
        # Convert to sets for easier manipulation if needed, 
        # but stick to bitmasks for speed.
        # We need to try adding edges not in edges.
        # And removing edges to allow adding more.
        
        current_edges = list(edges)
        current_adj = adj[:]
        current_count = len(current_edges)
        edge_set = set(current_edges)
        
        improved = True
        while improved:
            improved = False
            # Try to add any missing edge
            missing = [e for e in all_edges if e not in edge_set]
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))
                    edge_set.add((u, v))
                    current_count += 1
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, add two? Or remove one, add one?
                # If we remove one, we might be able to add two.
                # This is complex. Let's stick to single add/remove for now.
                # Or try to find a pair of edges to remove that allows 3 to be added?
                # For now, simple local search is better than nothing.
                pass
        
        return current_edges

    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 110.0
    
    rng = random.Random(42)
    
    # Run multiple greedy + local search
    for i in range(200):
        if time.time() - start_time > time_limit:
            break
            
        order = all_edges[:]
        rng.shuffle(order)
        
        edges, adj = greedy_construction(order)
        
        # Local search
        edges = local_search(edges, adj)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    # Verification
    def is_valid(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
        
    if not is_valid(best_edges):
        # Fallback
        return [[4 * k + a, 4 * k + b] for k in range(5) for a in range(4) for b in range(a + 1, 4)]
        
    return [[u, v] for u, v in best_edges]
