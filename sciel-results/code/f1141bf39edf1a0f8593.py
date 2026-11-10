<<<<<<< SEARCH
def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid_graph(n, edges):
        # Build adjacency bitsets
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check for regular induced subgraph of size >= 6
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                degs = []
                for u in subset:
                    deg = 0
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    deg = (adj[u] & mask).bit_count()
                    degs.append(deg)
                
                if len(set(degs)) == 1:
                    return False
        return True

    best_n = 5
    best_edges = []

    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
            
        found = False
        for seed in range(200):
            if time.time() > deadline - 2:
                break
                
            random.seed(seed)
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            if is_valid_graph(n, edges):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
=======
def find_violation(adj, n):
    """
    Find a subset of size >= 6 that induces a regular subgraph.
    Returns the subset as a tuple, or None if no violation found.
    """
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for u in subset:
                deg = (adj[u] & mask).bit_count()
                degs.append(deg)
            
            if len(set(degs)) == 1:
                return subset
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
            
        found = False
        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            random.seed(restart * 1000 + n)
            
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline - 2:
                    break
                
                viol = find_violation(adj, n)
                if viol is None:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                u, v = random.sample(list(viol), 2)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
        # If not found, try next n
        
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
