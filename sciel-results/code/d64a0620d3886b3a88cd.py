import time

# Public Task Contract data
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s safety margin out of 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    max_mask = 1 << n
    
    # Precompute validity: is the subset a clique or an independent set?
    # is_valid[mask] = True if mask induces a clique or an independent set
    is_valid = [False] * max_mask
    is_valid[0] = True # Empty set is trivially both, but we don't use it as a part
    
    for mask in range(1, max_mask):
        is_c = True
        is_i = True
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Check if v is connected to all other vertices in mask (for clique)
            # adj[v] & mask should be equal to mask ^ lsb (all others)
            if (adj[v] & mask) != (mask ^ lsb):
                is_c = False
            # Check if v is connected to no other vertices in mask (for independent set)
            # adj[v] & mask should be 0
            if (adj[v] & mask) != 0:
                is_i = False
            
            if not is_c and not is_i:
                break
                
            m ^= lsb
            
        if is_c or is_i:
            is_valid[mask] = True
            
    # Dynamic Programming to find minimum number of parts
    # dp[mask] = min number of valid parts to cover 'mask'
    INF = float('inf')
    dp = [INF] * max_mask
    dp[0] = 0
    parent = [-1] * max_mask # To reconstruct the partition
    
    # Iterate through all masks
    for mask in range(1, max_mask):
        if time.time() > deadline:
            # Fallback: greedy heuristic if DP takes too long (unlikely for n=14)
            # But we must return a valid partition.
            # For n=14, this branch is practically unreachable.
            pass
        
        # Iterate through all non-empty subsets of mask
        sub = mask
        best_val = INF
        best_sub = -1
        
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        parent[mask] = best_sub
        
    # Reconstruct the partition
    parts = []
    cur = full_mask
    while cur:
        sub = parent[cur]
        if sub == -1:
            # Fallback if parent not set (should not happen for n=14)
            # Add remaining vertices as singletons (which are always valid)
            remaining = []
            m = cur
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                remaining.append(int(v))
                m ^= lsb
            if remaining:
                parts.append(remaining)
            break
        
        part = []
        m = sub
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            part.append(int(v)) # Ensure Python int
            m ^= lsb
        parts.append(part)
        cur ^= sub
        
    # Final validation check to ensure no logic errors
    # 1. All vertices covered exactly once
    all_vertices = []
    for p in parts:
        all_vertices.extend(p)
    if sorted(all_vertices) != list(range(n)):
        # This should never happen, but if it does, return a trivial valid partition (singletons)
        return [[int(i)] for i in range(n)]
        
    # 2. Each part is a clique or independent set
    for p in parts:
        mask = 0
        for v in p:
            mask |= (1 << v)
        if not is_valid[mask]:
             # Invalid partition found, return singletons
             return [[int(i)] for i in range(n)]
             
    return parts

if __name__ == '__main__':
    res = construct()
    print(f"Number of parts: {len(res)}")
    print(f"Partition: {res}")
