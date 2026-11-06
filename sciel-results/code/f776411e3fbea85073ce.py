import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Branch and bound for independent set of size 10
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Heuristic: pick vertex with max degree in candidates
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            
            # Branch 2: Exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            
            return False
        
        return backtrack(mask_all, 0)

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    # Try to construct valid graphs for n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        while time.time() < deadline:
            adj = [0] * n
            
            # Generate random maximal triangle-free graph
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if valid
            if not has_independent_set_10(adj, n):
                return {'n': n, 'edges': build_edge_list(adj, n)}

    # Fallback: Andrasfai graph And(9)
    n = 27
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d == 1 or d == n - 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify And(9)
    if not has_independent_set_10(adj, n):
        return {'n': n, 'edges': build_edge_list(adj, n)}
    
    # Should not reach here
    return {'n': 27, 'edges': []}
