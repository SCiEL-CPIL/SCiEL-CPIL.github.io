import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def popcount(x):
        return bin(x).count('1')

    def find_violation_fast(adj, n):
        """
        Find a subset of size >= 6 that induces a regular subgraph.
        Returns (subset_mask, size) or (None, 0) if none found.
        Optimized to check smaller sizes first.
        """
        # Check sizes from 6 to n
        for k in range(6, n + 1):
            # Iterate over combinations of size k
            # For n=20, k=6 is ~38k combos. k=7 is ~77k.
            # This is much faster than 2^20 = 1M masks with popcount checks.
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for u in comb:
                    mask |= (1 << u)
                
                # Check regularity
                first_deg = -1
                is_regular = True
                for u in comb:
                    deg = popcount(adj[u] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_regular = False
                        break
                
                if is_regular:
                    return mask, k
        
        return None, 0

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def solve(n, time_budget):
        """
        Attempt to find a valid graph on n vertices within time_budget.
        """
        if time.time() - start_time > time_limit - time_budget:
            return None, None

        best_adj = None
        best_edges = None
        
        # Try multiple random restarts
        while time.time() - start_time < time_limit - time_budget:
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # We will perform a fixed number of steps or until no violation is found
            # Or until time runs out
            for _ in range(2000): # Max steps per restart
                if time.time() - start_time > time_limit - time_budget:
                    break
                
                mask, k = find_violation_fast(adj, n)
                
                if mask is None:
                    # Valid graph found
                    if best_adj is None or n > 0: # n is fixed in this call
                        best_adj = adj[:]
                        best_edges = get_edges_from_adj(adj, n)
                    return n, best_edges
                
                # Found violation. Flip a random edge within the violating subset.
                verts = [i for i in range(n) if (mask >> i) & 1]
                
                if len(verts) < 2:
                    break
                    
                u, v = random.sample(verts, 2)
                
                # Flip edge (u, v)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # If we exit the inner loop without finding a valid graph, we restart
            # The outer while loop handles restarts

        return None, None

    # Try n=21 first
    # Give it 100 seconds
    result = solve(21, 100.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Try n=20
    # Give it 40 seconds
    result = solve(20, 40.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Fallback: Return a trivially valid graph (n=5)
    # Any graph on 5 vertices is valid.
    adj = [0] * 5
    for i in range(5):
        for j in range(i + 1, 5):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 5, 'edges': get_edges_from_adj(adj, 5)}
