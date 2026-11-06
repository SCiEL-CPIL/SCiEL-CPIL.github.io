import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()
    
    # Target n values to try
    targets = [35, 34, 33, 32, 31, 30, 28, 25, 20]
    
    for n in targets:
        if time.time() > deadline:
            break
        
        # Try to find a valid graph for this n
        result = try_find_graph(n, deadline)
        if result:
            # Verify the result before accepting (using fast bitmask check)
            if is_valid_fast(result):
                best_result = result
                if n >= 35:
                    return best_result
                # If we found a good graph, we might still want to try larger n?
                # The loop is descending, so if 35 fails, we try 34.
                # If 34 succeeds, we return 34.
                # But if 35 was possible, we missed it.
                # Given the difficulty, finding 34 is a big win.
                # Let's return if n >= 34 to save time for verification?
                # Or continue? The loop will try 33 next.
                # We should return if we find a high n.
                return best_result

    return best_result

def is_valid_fast(graph):
    n = graph['n']
    edges = graph['edges']
    
    # Build bitmask adjacency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check K4
    # For each pair (i, j) with edge, check if their common neighbors contain an edge
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                # Common neighbors of i and j, excluding i and j
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                # Check if there is an edge in the subgraph induced by 'common'
                # Iterate through bits in common
                c = common
                while c:
                    k_bit = c & -c
                    k = k_bit.bit_length() - 1
                    # Check if k is connected to any other vertex in common
                    if adj[k] & (common ^ k_bit):
                        return False
                    c ^= k_bit
    
    # Check Independent 6-set (K6 in complement)
    # Complement adjacency
    comp_adj = [0] * n
    mask_all = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = mask_all ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i + 1, n):
            if comp_adj[i] & (1 << j):
                common = comp_adj[i] & comp_adj[j] & ~(1 << i) & ~(1 << j)
                c = common
                while c:
                    k_bit = c & -c
                    k = k_bit.bit_length() - 1
                    if comp_adj[k] & (common ^ k_bit):
                        return False
                    c ^= k_bit
                    
    return True

def try_find_graph(n, deadline):
    # Local Search with Bitmasks
    # Initialize
    adj = [0] * n
    mask_all = (1 << n) - 1
    
    # Random initialization
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Local Search
    # We will sample random violations to fix
    # To avoid getting stuck, we track the number of "failed" samples
    # If we sample many times without finding a violation, we might be valid?
    # No, we must be sure. We use is_valid_fast at the end.
    # But if we are stuck in a local minimum with violations, we need to perturb.
    
    max_iters = 20000
    iters = 0
    
    for restart in range(3):
        if time.time() > deadline:
            return None
            
        # Re-initialize if restart > 0
        if restart > 0:
             for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            iters += 1
            
            # Try to find a K4 violation
            found_k4 = False
            for _ in range(100): # Sample 100 times
                i, j, k, l = random.sample(range(n), 4)
                if adj[i] & (1<<j) and adj[i] & (1<<k) and adj[i] & (1<<l) and \
                   adj[j] & (1<<k) and adj[j] & (1<<l) and \
                   adj[k] & (1<<l):
                    # Found K4 (i,j,k,l)
                    # Pick a random edge to flip
                    edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                    u, v = random.choice(edges_in_k4)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    found_k4 = True
                    break
                    
            if found_k4:
                continue
                
            # Try to find an I6 violation
            found_i6 = False
            for _ in range(50): # Sample 50 times (slower to find)
                i, j, k, l, m, p = random.sample(range(n), 6)
                # Check if independent: no edges between any pair
                if not (adj[i] & (1<<j) or adj[i] & (1<<k) or adj[i] & (1<<l) or 
                        adj[i] & (1<<m) or adj[i] & (1<<p) or
                        adj[j] & (1<<k) or adj[j] & (1<<l) or 
                        adj[j] & (1<<m) or adj[j] & (1<<p) or
                        adj[k] & (1<<l) or adj[k] & (1<<m) or 
                        adj[k] & (1<<p) or
                        adj[l] & (1<<m) or adj[l] & (1<<p) or
                        adj[m] & (1<<p)):
                    # Found I6
                    edges_in_i6 = [(i,j), (i,k), (i,l), (i,m), (i,p),
                                   (j,k), (j,l), (j,m), (j,p),
                                   (k,l), (k,m), (k,p),
                                   (l,m), (l,p),
                                   (m,p)]
                    u, v = random.choice(edges_in_i6)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    found_i6 = True
                    break
                    
            if found_i6:
                continue
                
            # If no violation found in sampling, it might be valid or we missed it.
            # We can break out and verify.
            # But to be safe, let's do a few more iterations or verify.
            # Given the sampling rates, if we miss a K4, it's rare (if many exist).
            # If we miss an I6, it's possible.
            # Let's verify every 1000 iterations? Or just break and verify.
            break
            
        # Construct result and verify
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
                    
        result = {'n': n, 'edges': edges}
        if is_valid_fast(result):
            return result
            
    return None
