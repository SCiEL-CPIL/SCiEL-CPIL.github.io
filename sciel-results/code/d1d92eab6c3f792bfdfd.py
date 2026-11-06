import random
import time

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_valid_addition(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
        
        # 2. For any w in N(v) \ {u}, v becomes a common neighbor of u and w.
        #    So |N(u) ∩ N(w)| must currently be < 2.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            common_uw = adj[u] & adj[w]
            if popcount(common_uw) >= 2:
                return False
            neighbors_v ^= lsb
            
        # 3. Similarly for w in N(u) \ {v}
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            common_vw = adj[v] & adj[w]
            if popcount(common_vw) >= 2:
                return False
            neighbors_u ^= lsb
            
        return True

    def get_missing_edges(adj):
        """Return list of missing edges."""
        missing = []
        for i in range(n):
            mask = ~adj[i] & ((1 << n) - 1) & ~((1 << i) - 1)
            mask &= ~((1 << (i + 1)) - 1)  # Only j > i
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                missing.append((i, j))
                mask ^= lsb
        return missing

    def try_add_all(adj):
        """Try to add all possible missing edges until no more can be added."""
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    return False
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return True

    def edge_count(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def extract_edges(adj):
        current_edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                if i < j:
                    current_edges.append([i, j])
                mask ^= lsb
        return current_edges

    best_edges = []
    best_count = 0
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Initialize adjacency bitmasks
        adj = [0] * n
        
        # Greedy construction with random edge order
        edges = all_pairs[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Hill climbing: add all possible edges
        try_add_all(adj)
        
        # Swap-based local search: try removing one edge and adding two others
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            current_count = edge_count(adj)
            
            # Get current edges
            current_edges = extract_edges(adj)
            random.shuffle(current_edges)
            
            for rem_u, rem_v in current_edges:
                if time.time() > deadline:
                    break
                
                # Temporarily remove edge (rem_u, rem_v)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                
                # Try to add all possible edges
                added = []
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added.append((u, v))
                
                new_count = edge_count(adj)
                
                if new_count > current_count:
                    # Keep the change
                    improved = True
                else:
                    # Revert
                    for u, v in added:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
        
        # Final hill climbing
        try_add_all(adj)
        
        # Update best
        count = edge_count(adj)
        if count > best_count:
            best_count = count
            best_edges = extract_edges(adj)
        
        # If we've found a very dense graph, we can be more aggressive
        # with time allocation for remaining restarts
        if best_count >= 95 and time.time() > start_time + 80:
            break
            
    return best_edges
