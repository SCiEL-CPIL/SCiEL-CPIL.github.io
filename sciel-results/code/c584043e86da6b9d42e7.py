import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find graphs with larger n
    for n in [20, 18, 16, 15, 12, 10]:
        if time.time() > deadline - 20.0:
            break
            
        # Generate list of bad subset masks to check
        # We only need to check subsets of size >= 6
        subset_masks = []
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                subset_masks.append(mask)
        
        # Local Search
        # Initialize random graph
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        def count_bad(adj):
            bad = 0
            for mask in subset_masks:
                degs = []
                for i in range(n):
                    if mask & (1 << i):
                        degs.append((adj[i] & mask).bit_count())
                if len(set(degs)) == 1:
                    bad += 1
            return bad
        
        initial_bad = count_bad(adj_masks)
        
        if initial_bad == 0:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_masks[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        best_bad = initial_bad
        best_adj = adj_masks.copy()
        
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        current_adj = adj_masks
        current_bad = initial_bad
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            # Calculate new bad count
            # To save time, we can just do a full check every few iterations
            # Or just do full check every time? 
            # 1 full check ~ 1-2s. 5000 iters is too many.
            # Let's do full check every 50 iters?
            # No, we need to know if it got better.
            # Let's try a cheaper heuristic:
            # Only check subsets containing u and v?
            # Number of such subsets is ~ 2^(n-2).
            # For n=20, 2^18 = 262k.
            # Checking 262k subsets is ~ 1/4 of full check.
            # Let's do this.
            
            # Identify subsets containing u and v
            # We can't easily filter subset_masks without iterating.
            # Let's just do a full check every 100 iterations to update best,
            # and use a random acceptance criterion otherwise?
            # No, that's not local search.
            
            # Let's optimize count_bad by only checking affected subsets.
            # We need a list of masks containing both u and v.
            # Precomputing this for all pairs is memory heavy.
            # Let's just iterate through subset_masks and check if (mask & (1<<u) and mask & (1<<v)).
            # This is 10^6 checks per move. Too slow.
            
            # Alternative:
            # Just do full check every K iterations.
            # And use a "step" that is likely to improve?
            # No, let's just do full check every iteration but with a smaller n first?
            
            # Actually, for n=10, full check is fast (386 subsets).
            # For n=15, full check is ~15k subsets.
            # For n=20, full check is ~1M subsets.
            
            # Let's adjust strategy based on n.
            if n <= 15:
                new_bad = count_bad(current_adj)
                delta = new_bad - current_bad
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_bad = new_bad
                    if new_bad < best_bad:
                        best_bad = new_bad
                        best_adj = current_adj.copy()
                        if best_bad == 0:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if best_adj[i] & (1 << j):
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
                else:
                    # Revert
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                T *= cooling
            else:
                # For n > 15, full check is slow.
                # Do full check every 50 iterations.
                if it % 50 == 0:
                    new_bad = count_bad(current_adj)
                    delta = new_bad - current_bad
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_bad = new_bad
                        if new_bad < best_bad:
                            best_bad = new_bad
                            best_adj = current_adj.copy()
                            if best_bad == 0:
                                edges = []
                                for i in range(n):
                                    for j in range(i + 1, n):
                                        if best_adj[i] & (1 << j):
                                            edges.append([i, j])
                                return {'n': n, 'edges': edges}
                    else:
                        # Revert
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)
                    T *= cooling
                else:
                    # Just accept the move to explore, don't check
                    pass
            
        # After search, if we have a valid graph, return it
        if best_bad == 0:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: return seed
    return {'n': 5, 'edges': []}
