import random
import time
from itertools import combinations

def is_valid_graph(n, adj, precomputed_masks):
    """
    adj: list of integers where adj[u] is a bitmask of neighbors of u.
    precomputed_masks: dict mapping size -> list of (mask, tuple_of_vertices)
    Returns True if no subset of size >= 6 induces a regular subgraph.
    """
    # Check 6-vertex subsets first
    for mask, subset in precomputed_masks.get(6, []):
        u = subset[0]
        target_deg = (adj[u] & mask).bit_count()
        
        valid = True
        for v in subset[1:]:
            deg_v = (adj[v] & mask).bit_count()
            if deg_v != target_deg:
                valid = False
                break
        
        if valid:
            return False
    
    # Check 7-vertex subsets
    for mask, subset in precomputed_masks.get(7, []):
        u = subset[0]
        target_deg = (adj[u] & mask).bit_count()
        
        valid = True
        for v in subset[1:]:
            deg_v = (adj[v] & mask).bit_count()
            if deg_v != target_deg:
                valid = False
                break
        
        if valid:
            return False
    
    # Check 8+ vertex subsets
    for size in range(8, n + 1):
        for mask, subset in precomputed_masks.get(size, []):
            u = subset[0]
            target_deg = (adj[u] & mask).bit_count()
            
            valid = True
            for v in subset[1:]:
                deg_v = (adj[v] & mask).bit_count()
                if deg_v != target_deg:
                    valid = False
                    break
            
            if valid:
                return False
    
    return True

def construct():
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Precompute masks for each n
    mask_cache = {}
    
    for n in [21, 20]:
        if n not in mask_cache:
            precomputed = {}
            for size in range(6, n + 1):
                entries = []
                for subset in combinations(range(n), size):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    entries.append((mask, subset))
                precomputed[size] = entries
            mask_cache[n] = precomputed
        
        precomputed = mask_cache[n]
        found = None
        
        # Try different probabilities
        for p in [0.5, 0.4, 0.6]:
            while time.time() - start_time < time_limit and found is None:
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if is_valid_graph(n, adj, precomputed):
                    found = adj
                    break
                
                if time.time() - start_time > time_limit - 5:
                    break
        
        if found:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (found[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: return a trivial valid graph
    return {'n': 5, 'edges': []}
