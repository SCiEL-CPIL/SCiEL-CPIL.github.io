import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def check_regular_fast(adj, n):
        """Check if any subset of size >= 6 induces a regular subgraph.
        Returns (is_valid, bad_count) where is_valid is True if no bad subsets exist."""
        # For each subset of size >= 6, check if all vertices have same degree in induced subgraph
        # This is expensive, so we use early termination
        
        # First, check all 6-vertex subsets (most likely to be regular)
        from itertools import combinations
        for size in range(6, n + 1):
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                # Check if induced subgraph on mask is regular
                degs = set()
                for v in combo:
                    deg = (adj[v] & mask).bit_count()
                    degs.add(deg)
                    if len(degs) > 1:
                        break
                if len(degs) <= 1:
                    return False, 1  # Found a bad subset
        return True, 0
    
    def count_bad_subsets(adj, n):
        """Count all bad subsets (size >= 6 that induce regular subgraph)."""
        from itertools import combinations
        count = 0
        for size in range(6, n + 1):
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                degs = set()
                regular = True
                for v in combo:
                    deg = (adj[v] & mask).bit_count()
                    if deg not in degs:
                        degs.add(deg)
                        if len(degs) > 1:
                            regular = False
                            break
                if regular:
                    count += 1
        return count
    
    def try_random_graph(n, seed):
        """Try a random graph with given seed, return (adj, bad_count)."""
        random.seed(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        bad = count_bad_subsets(adj, n)
        return adj, bad
    
    def local_search(adj, n, max_steps=500):
        """Local search to reduce bad subsets."""
        best_adj = adj[:]
        best_bad = count_bad_subsets(adj, n)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if best_bad == 0:
                break
            
            # Find a random bad subset and flip an edge within it
            # For efficiency, just try random edge flips and keep improvements
            improved = False
            for _ in range(20):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge (u, v)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_bad = count_bad_subsets(adj, n)
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_adj = adj[:]
                    improved = True
                    break
                else:
                    # Revert
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not improved and step > 100:
                break
        
        return best_adj, best_bad
    
    # Strategy: Try n=20 with many random restarts
    n = 20
    best_adj = None
    best_bad = float('inf')
    
    # Try multiple random graphs
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        adj, bad = try_random_graph(n, seed)
        if bad < best_bad:
            best_bad = bad
            best_adj = adj[:]
            if bad == 0:
                break
        
        # If we found a good candidate, do local search
        if bad < 50 and time.time() < deadline - 20:
            adj_ls, bad_ls = local_search(adj, n, max_steps=200)
            if bad_ls < best_bad:
                best_bad = bad_ls
                best_adj = adj_ls
                if bad_ls == 0:
                    break
    
    # If we found a valid graph, return it
    if best_bad == 0 and best_adj is not None:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Fallback: Try smaller n
    for n_try in [19, 18, 17, 16, 15]:
        if time.time() > deadline - 5:
            break
        for seed in range(50):
            if time.time() > deadline - 3:
                break
            adj, bad = try_random_graph(n_try, seed)
            if bad == 0:
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
    
    # Last resort: return a small valid graph
    return {'n': 5, 'edges': []}
