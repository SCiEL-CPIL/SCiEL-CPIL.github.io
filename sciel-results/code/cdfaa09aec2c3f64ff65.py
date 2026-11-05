import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def is_triangle_free_circulant(n, S):
        S_list = list(S)
        for a in S_list:
            for b in S_list:
                if a == b:
                    continue
                diff = (a - b) % n
                dist = diff if diff <= n - diff else n - diff
                if dist in S:
                    return False
        return True

    def has_independent_set_of_size_k(n, S, k):
        # Returns True if there is an independent set of size k in G
        # Equivalent to clique of size k in complement
        # Build complement adjacency list
        comp_adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = (j - i) % n
                dist = dist if dist <= n - dist else n - dist
                if dist not in S:
                    comp_adj[i].append(j)
                    comp_adj[j].append(i)

        # Simple DFS to find clique of size k
        # Use bitmasks for efficiency
        # n <= 40, so bitmask fits in int
        
        # Order vertices by degree in complement (descending) to prune better?
        # Actually, for bitmask, order doesn't matter for correctness, but for pruning.
        
        def dfs(candidates_mask, depth):
            if depth == k:
                return True
            if candidates_mask == 0:
                return False
            # Pruning
            if bin(candidates_mask).count('1') < k - depth:
                return False
            
            # Pick a vertex
            # We can pick any vertex. Let's pick the first one.
            v = -1
            m = candidates_mask
            idx = 0
            while m:
                if m & 1:
                    v = idx
                    break
                m >>= 1
                idx += 1
            
            if v == -1:
                return False
            
            # Remove v from candidates
            new_mask = candidates_mask & ~(1 << v)
            
            # Filter candidates to only those adjacent to v
            # adj_mask[v] is bitmask of neighbors of v
            adj_mask = 0
            for u in comp_adj[v]:
                adj_mask |= (1 << u)
            
            new_mask &= adj_mask
            
            return dfs(new_mask, depth + 1)

        init_mask = (1 << n) - 1
        return dfs(init_mask, 0)

    # Try n=39
    n = 39
    half = n // 2
    
    # Iterate masks from 2^19 - 1 down to 0
    # This prioritizes larger S (more edges)
    for mask in range(2**half - 1, -1, -1):
        if time.time() > deadline:
            break
        
        S = set()
        m = mask
        idx = 0
        while m:
            if m & 1:
                S.add(idx + 1)
            m >>= 1
            idx += 1
        
        if not S:
            continue
            
        if not is_triangle_free_circulant(n, S):
            continue
        
        # Check alpha < 10
        # If has_independent_set_of_size_k returns True, alpha >= 10, invalid.
        if has_independent_set_of_size_k(n, S, 10):
            continue
        
        # Valid graph found
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = (j - i) % n
                dist = dist if dist <= n - dist else n - dist
                if dist in S:
                    edges.append([i, j])
        
        return {'n': n, 'edges': edges}

    # If n=39 not found, try n=38
    n = 38
    half = n // 2
    for mask in range(2**half - 1, -1, -1):
        if time.time() > deadline:
            break
        
        S = set()
        m = mask
        idx = 0
        while m:
            if m & 1:
                S.add(idx + 1)
            m >>= 1
            idx += 1
        
        if not S:
            continue
            
        if not is_triangle_free_circulant(n, S):
            continue
        
        if has_independent_set_of_size_k(n, S, 10):
            continue
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = (j - i) % n
                dist = dist if dist <= n - dist else n - dist
                if dist in S:
                    edges.append([i, j])
        
        return {'n': n, 'edges': edges}

    # Fallback to seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
