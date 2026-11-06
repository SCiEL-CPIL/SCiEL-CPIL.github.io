import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_k3(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    count += popcount(adj[i] & adj[j] & ((1 << n) - 1))
        return count // 3 # Each triangle counted 3 times? No.
        # Let's just sum over i<j<k.
        # The above loop counts each triangle 3 times (once for each vertex as 'i'? No, i<j).
        # If i<j and edge exists, popcount(adj[i]&adj[j]) counts common neighbors k.
        # k can be > j or < i.
        # To count each triangle once, we need k > j.
        # So mask with bits > j.
        
    def count_k3_fast(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Count common neighbors k > j
                    mask = ((1 << n) - 1) ^ ((1 << (j+1)) - 1)
                    count += popcount(adj[i] & adj[j] & mask)
        return count

    def count_i3_fast(adj, n):
        # Count independent sets of size 3.
        # Equivalent to triangles in complement.
        # Complement adjacency: ~adj[i] & mask
        count = 0
        mask = (1 << n) - 1
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    # Common non-neighbors k > j
                    non_neigh_i = ~adj[i] & mask
                    non_neigh_j = ~adj[j] & mask
                    mask_k = mask ^ ((1 << (j+1)) - 1)
                    count += popcount(non_neigh_i & non_neigh_j & mask_k)
        return count

    def is_valid(adj, n):
        # Check K4
        mask = (1 << n) - 1
        # K4: i<j<k<l, all edges present.
        for i in range(n-3):
            for j in range(i+1, n-2):
                if not ((adj[i] >> j) & 1): continue
                for k in range(j+1, n-1):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1): continue
                    for l in range(k+1, n):
                        if not ((adj[i] >> l) & 1) or not ((adj[j] >> l) & 1) or not ((adj[k] >> l) & 1):
                            continue
                        return False
        # Check I6
        # This is slow. But we only call it on candidates.
        # We can check if complement has K6.
        # For now, assume if K3 count is low, I6 is unlikely?
        # No, we must be exact.
        # I will implement a backtracking K6 finder in complement.
        
        def has_k6_in_complement(adj, n):
            # Vertices 0..n-1.
            # We want 6 vertices with no edges between them in G.
            # i.e. all edges present in ~G.
            # Backtracking:
            # candidates: list of vertices that can be part of the independent set.
            # Start with all vertices.
            # Pick v. Add to set. Remove v and its neighbors in G (which are non-neighbors in ~G? No.)
            # In ~G, we need a clique.
            # Neighbors in ~G are non-neighbors in G.
            # So if we pick v, the next vertices must be non-neighbors of v in G.
            
            def bt(candidates, depth):
                if depth == 6:
                    return True
                if len(candidates) < depth:
                    return False
                # Try each candidate
                for idx in range(len(candidates)):
                    v = candidates[idx]
                    # New candidates: those in candidates[idx+1:] that are non-neighbors of v in G
                    new_cands = []
                    for u in candidates[idx+1:]:
                        if not ((adj[v] >> u) & 1):
                            new_cands.append(u)
                    if bt(new_cands, depth+1):
                        return True
                return False
            
            return bt(list(range(n)), 0)
            
        if has_k6_in_complement(adj, n):
            return False
            
        return True

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    best_n = 17
    best_adj = None
    # Initialize with Paley 17
    # squares = {x*x % 17 for x in range(1, 17)}
    # adj17 = [0]*17
    # for i in range(17):
    #   for j in range(i+1, 17):
    #     if (j-i) % 17 in squares:
    #       adj17[i] |= (1<<j)
    #       adj17[j] |= (1<<i)
    # best_adj = adj17

    # Try to find better
    for target_n in [35, 30, 25, 20]:
        if time.time() > deadline - 20:
            break
            
        # Random init
        p = 0.4
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i+1, target_n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_k3_fast(adj, target_n) + count_i3_fast(adj, target_n)
        best_local_cost = current_cost
        best_local_adj = adj[:]
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 20:
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v: continue
            
            # Flip edge
            if (adj[u] >> v) & 1:
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Delta cost
                # K3: lose triangles containing (u,v)
                # I3: gain independent sets containing (u,v)
                # Recalculate cost? Or delta?
                # Delta is safer if implemented correctly, but recalc is O(n^2).
                # n=35, n^2=1225. Very fast.
                new_cost = count_k3_fast(adj, target_n) + count_i3_fast(adj, target_n)
                delta = new_cost - current_cost
            else:
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_cost = count_k3_fast(adj, target_n) + count_i3_fast(adj, target_n)
                delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local_adj = adj[:]
                    # Check validity occasionally
                    if time.time() < deadline - 25:
                        if is_valid(best_local_adj, target_n):
                            # Found valid graph!
                            return {'n': target_n, 'edges': to_edges(best_local_adj, target_n)}
            else:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            T *= cooling
        
        # End of SA for this n
        if best_local_cost < 100: # Heuristic: if cost is low, check validity
             if is_valid(best_local_adj, target_n):
                 return {'n': target_n, 'edges': to_edges(best_local_adj, target_n)}
        
        if target_n > best_n:
             # Even if invalid, we might have a valid subgraph? No.
             # We only update best if valid.
             pass

    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': to_edges(adj, 17)}
