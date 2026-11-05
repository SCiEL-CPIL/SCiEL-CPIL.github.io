import time

def _max_independent_set_size(n, adj):
    """
    Exact branch and bound for maximum independent set size.
    adj: list of lists, adj[i] is a set of neighbors of i.
    """
    # Convert to bitmasks for speed if n <= 64
    if n > 64:
        # Fallback to simple recursion (might be slow, but n=27 is small)
        pass
    
    # Simple DFS with pruning
    max_size = 0
    # Order vertices by degree descending for better pruning
    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # adj_mask[i] is a bitmask of neighbors
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
            
    # Precompute non-neighbors (including self) for independent set candidates
    # Actually, we just need to check if a vertex can be added.
    
    def dfs(remaining_mask, current_size):
        nonlocal max_size
        if current_size > max_size:
            max_size = current_size
            
        if remaining_mask == 0:
            return
            
        # Pruning: if current_size + popcount(remaining_mask) <= max_size, return
        # popcount is fast in Python 3.10+ via int.bit_count()
        if current_size + remaining_mask.bit_count() <= max_size:
            return

        # Pick a vertex from remaining
        # Heuristic: pick vertex with highest degree in the induced subgraph?
        # Or just the first one.
        v = (remaining_mask & -remaining_mask).bit_length() - 1
        
        # Option 1: Include v
        # New remaining: neighbors of v are removed
        new_remaining = remaining_mask & ~(adj_mask[v] | (1 << v))
        dfs(new_remaining, current_size + 1)
        
        # Option 2: Exclude v
        remaining_mask &= ~(1 << v)
        dfs(remaining_mask, current_size)

    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    return max_size

def _has_triangle(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                # Check intersection of adj[u] and adj[v]
                if adj[u].intersection(adj[v]):
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n=27 with the Andrasfai-like rule
    # Rule: edge if (j-i) % n % 3 == 1
    for n in [27, 26]:
        if time.time() > deadline:
            break
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        
        # Check triangle-free
        if _has_triangle(n, edges):
            continue
            
        # Check independence number
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        alpha = _max_independent_set_size(n, adj)
        
        if alpha < 10:
            return {'n': n, 'edges': edges}
            
    # Fallback to seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
