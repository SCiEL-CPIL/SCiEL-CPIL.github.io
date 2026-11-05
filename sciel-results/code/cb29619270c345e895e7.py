import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    def check_c4_add(adj, u, v):
        # Check if adding (u, v) creates a C4
        # Need to check if any a in N(u) is adjacent to any b in N(v)
        for a in adj[u]:
            if adj[a] & adj[v]:
                return False
        return True

    def verify_alpha_lt_11(n, adj, time_limit=1.0):
        # Returns True if alpha < 11, False if alpha >= 11 or timeout
        # Uses Branch and Bound
        # adj is list of sets
        
        # Convert to bitmasks for speed
        # n <= 39, so we can use Python integers as bitsets
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_masks[u] = mask
            
        # Greedy upper bound on independent set size from a set of candidates
        def greedy_upper_bound(candidates_mask):
            count = 0
            remaining = candidates_mask
            while remaining:
                # Pick vertex with min degree in remaining
                min_deg = float('inf')
                min_v = -1
                # Iterate through set bits
                r = remaining
                while r:
                    lsb = r & -r
                    v = lsb.bit_length() - 1
                    # Count neighbors in remaining
                    # neighbors = adj_masks[v] & remaining
                    # deg = bin(neighbors).count('1')
                    deg = bin(adj_masks[v] & remaining).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    r ^= lsb
                count += 1
                # Remove min_v and its neighbors
                remaining &= ~(1 << min_v)
                remaining &= ~adj_masks[min_v]
            return count

        best_alpha = 0
        # State: (current_set_mask, candidates_mask, current_size)
        # We want to find if there is a set of size 11
        # If we find one, return False
        # If we exhaust search and max size < 11, return True
        
        import heapq
        # Simple DFS with pruning
        # Order vertices by degree?
        
        def dfs(candidates_mask, current_size, current_mask):
            nonlocal best_alpha
            if current_size + greedy_upper_bound(candidates_mask) <= best_alpha:
                return False # Pruned, but we need to know if we found 11
            
            if current_size >= 11:
                return True # Found a set of size 11
            
            if not candidates_mask:
                if current_size > best_alpha:
                    best_alpha = current_size
                return False
            
            # Pick a vertex
            # Heuristic: Pick vertex with max degree in candidates?
            # Or min degree?
            # Min degree is good for packing.
            min_deg = float('inf')
            min_v = -1
            r = candidates_mask
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = bin(adj_masks[v] & candidates_mask).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                r ^= lsb
            
            # Branch 1: Include min_v
            # New candidates: candidates without min_v and its neighbors
            new_candidates = candidates_mask & ~(1 << min_v) & ~adj_masks[min_v]
            if dfs(new_candidates, current_size + 1, current_mask | (1 << min_v)):
                return True
            
            # Branch 2: Exclude min_v
            # New candidates: candidates without min_v
            new_candidates = candidates_mask & ~(1 << min_v)
            if dfs(new_candidates, current_size, current_mask):
                return True
                
            return False

        # Initial candidates: all vertices
        all_mask = (1 << n) - 1
        
        # Set a time limit for the verification
        v_start = time.time()
        
        # To prevent stack overflow or infinite loops, we rely on the bound
        # For n=39, recursion depth is <= 39.
        
        # We want to know if alpha >= 11.
        # If dfs returns True, alpha >= 11.
        # If dfs returns False, alpha < 11 (assuming no timeout).
        
        # However, standard DFS might be slow.
        # Let's add a timeout check inside dfs?
        # Python recursion doesn't easily support timeout.
        # We can use a global flag.
        
        timeout_flag = [False]
        
        def dfs_timed(candidates_mask, current_size, current_mask):
            nonlocal best_alpha
            if time.time() - v_start > time_limit:
                timeout_flag[0] = True
                return False # Assume not found yet, but flag timeout
            
            if current_size + greedy_upper_bound(candidates_mask) < 11:
                return False # Cannot reach 11
            
            if current_size >= 11:
                return True
            
            if not candidates_mask:
                return False
            
            min_deg = float('inf')
            min_v = -1
            r = candidates_mask
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = bin(adj_masks[v] & candidates_mask).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                r ^= lsb
            
            # Include min_v
            new_candidates = candidates_mask & ~(1 << min_v) & ~adj_masks[min_v]
            if dfs_timed(new_candidates, current_size + 1, current_mask | (1 << min_v)):
                return True
            
            # Exclude min_v
            new_candidates = candidates_mask & ~(1 << min_v)
            if dfs_timed(new_candidates, current_size, current_mask):
                return True
                
            return False

        found_11 = dfs_timed(all_mask, 0, 0)
        
        if timeout_flag[0]:
            return False # Reject due to uncertainty
        
        if found_11:
            return False # Alpha >= 11
        
        return True # Alpha < 11

    def try_construct(n, max_restarts=50):
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
                
            adj = [set() for _ in range(n)]
            edges = []
            
            possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                if check_c4_add(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Verify
            if verify_alpha_lt_11(n, adj, time_limit=2.0):
                return edges
                
        return None

    # Try n from 39 down to 31
    for n in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        edges = try_construct(n, max_restarts=20)
        
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
