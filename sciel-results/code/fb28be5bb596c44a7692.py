import random
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 115  # 5s margin
    
    best_n = 5
    best_edges = []
    
    # We will try n from 20 down to 6
    # If we find a valid graph, we update best
    # We can stop if we find n=20? 
    # The task says maximize n. 
    # If we find 20, we should try to see if 21 is possible?
    # But 21 is very hard. 
    # Let's focus on getting a high n reliably.
    # We will try 20, then 19, etc.
    # If we find a valid graph for n, we save it.
    # We can continue to search for larger n if time permits?
    # No, if we find n=20, it's likely the max.
    # But maybe we can find n=21?
    # Let's try 21 first? 
    # Probability of finding 21 is low.
    # Probability of finding 20 is higher.
    # Let's try 20 first. If we find it, we return it.
    # If we don't find it in 10s, we try 19.
    
    for n in range(20, 5, -1):
        # Precompute masks for this n
        # We only need masks for sizes 6 to n
        # To save time, we generate them lazily or cache per n?
        # Caching per n is good if we do multiple restarts.
        
        masks_by_size = {}
        for k in range(6, n + 1):
            masks = []
            for combo in itertools.combinations(range(n), k):
                m = 0
                for v in combo:
                    m |= (1 << v)
                masks.append(m)
            masks_by_size[k] = masks
        
        # Try a few restarts
        # Time budget per n: e.g., 10s
        n_start = time.time()
        
        for restart in range(10):
            if time.time() - start_time > time_limit:
                return {'n': best_n, 'edges': best_edges}
            if time.time() - n_start > 10:
                break
            
            # Init random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # We want to minimize violations.
            # Start with checking size 6.
            # If 0, check 7, etc.
            
            while time.time() - start_time < time_limit:
                # Find violations
                # Check sizes in increasing order
                violations = []
                for k in range(6, n + 1):
                    for m in masks_by_size[k]:
                        # Check if regular
                        # Get vertices
                        # Optimization: 
                        # We can just check if all degrees are equal.
                        # degs = []
                        # for v in range(n):
                        #    if m & (1 << v):
                        #       degs.append((adj[v] & m).bit_count())
                        # if len(set(degs)) == 1:
                        #    violations.append((k, m))
                        
                        # Faster check:
                        # Count vertices in m
                        # If m is 0, skip (not possible for k>=6)
                        
                        # Let's use a helper
                        is_reg = True
                        first_deg = -1
                        count = 0
                        for v in range(n):
                            if m & (1 << v):
                                d = (adj[v] & m).bit_count()
                                if first_deg == -1:
                                    first_deg = d
                                elif d != first_deg:
                                    is_reg = False
                                    break
                                count += 1
                        if is_reg and count == k: # count should be k
                            violations.append((k, m))
                            # We can break early if we have enough violations?
                            # No, we want to fix them all eventually.
                            # But for move selection, we just need one.
                            # However, if we have many, we might pick a "bad" one.
                            # Let's collect a few.
                            if len(violations) > 10:
                                break
                    if len(violations) > 10:
                        break
                
                if not violations:
                    # Valid graph found!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    
                    # If n == 20, we can return immediately?
                    # Or try to improve? No, 20 is max known.
                    # But maybe 21 is possible?
                    # Let's return 20 to be safe and fast.
                    # The task says "Maximize n".
                    # If we find 20, it's a strong result.
                    # We can try to run for a bit longer to see if we can find 21?
                    # No, 21 is a different search space.
                    # Let's return 20.
                    return {'n': n, 'edges': edges}
                
                # Pick a random violation
                k, m = random.choice(violations)
                
                # Pick an edge to flip
                # Prefer edges inside the violation
                verts_in_m = [v for v in range(n) if m & (1 << v)]
                
                # Try to flip an edge inside m
                if len(verts_in_m) >= 2:
                    # Pick two random vertices in m
                    u, v = random.sample(verts_in_m, 2)
                    # Flip edge (u, v)
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Should not happen for k>=6
                    pass
    
    return {'n': best_n, 'edges': best_edges}
