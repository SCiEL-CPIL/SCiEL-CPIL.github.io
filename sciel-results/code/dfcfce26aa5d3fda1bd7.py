import time
import random
import itertools

DEADLINE = time.time() + 118

def _check_violations(adj, n):
    """
    Returns a list of violating subsets (size >= 6).
    adj: list of ints (bitmasks).
    """
    violations = []
    # Precompute masks for subsets? No, generate on fly.
    # For speed, we can check size 6 first, then 7, etc.
    # If we find many, we can stop early? No, we need to know if it's 0.
    # But for local search, we just need *one* or a few to fix.
    
    # Optimization: Check only subsets of size 6?
    # If there is a regular induced subgraph of size > 6, it contains regular induced subgraphs of size 6?
    # NO. A 7-regular graph (degree 6) on 7 vertices is $K_7$. $K_7$ contains $K_6$ which is regular.
    # What about $C_7$? Not regular induced subgraph of size 7? $C_7$ is regular (degree 2).
    # Does $C_7$ contain a regular induced subgraph of size 6?
    # Induced subgraph of $C_7$ on 6 vertices is a path $P_6$. $P_6$ is NOT regular.
    # So checking only size 6 is NOT sufficient.
    # We must check all sizes.
    
    # However, if we find a violation of size 6, we can fix it.
    # If we fix all size 6 violations, do size 7+ disappear? Not necessarily.
    # But size 6 is the most frequent.
    
    # Let's just check all.
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            # Check if regular
            # Mask for subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check degrees
            degs = []
            for v in subset:
                # Count neighbors of v within subset
                d = (adj[v] & mask).bit_count()
                degs.append(d)
            
            if all(d == degs[0] for d in degs):
                violations.append(subset)
                # Early exit for local search? 
                # If we just want to find *any* violation to fix, we can return early.
                # But we want to minimize count.
                # For now, collect a few.
                if len(violations) >= 10:
                    return violations
    return violations

def construct():
    start_time = time.time()
    best_n = 5
    best_edges = []
    
    # Try to build graphs of increasing size
    for n in range(10, 23):
        if time.time() > DEADLINE - 10:
            break
            
        found = False
        # Increase restarts for larger n? Or decrease?
        # Larger n is harder.
        max_restarts = 50 if n <= 15 else 20
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local Search
            # We will try to reduce the number of violations.
            # We track the current violations list.
            violations = _check_violations(adj, n)
            
            # If no violations, we found one!
            if not violations:
                found = True
                break
            
            # If violations, try to fix them
            # We do a local search loop
            stuck_count = 0
            for step in range(2000):
                if time.time() > DEADLINE - 5:
                    break
                
                if not violations:
                    found = True
                    break
                    
                # Pick a random violation
                v_sub = random.choice(violations)
                
                # Try flipping edges within v_sub
                improved = False
                # Shuffle edges in v_sub
                edges_in_sub = list(itertools.combinations(v_sub, 2))
                random.shuffle(edges_in_sub)
                
                for u, v in edges_in_sub:
                    # Flip edge u-v
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    new_violations = _check_violations(adj, n)
                    
                    if len(new_violations) < len(violations):
                        violations = new_violations
                        improved = True
                        if not violations:
                            found = True
                            break
                    else:
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    if found:
                        break
                
                if found:
                    break
                
                if not improved:
                    # Random perturbation
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        violations = _check_violations(adj, n)
                
                if time.time() > DEADLINE - 5:
                    break
        
        if found:
            best_n = n
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            best_edges = edges
            # We can stop here if we reached 22, or continue to try for higher?
            # Task says maximize n. So if we found 20, we should try 21, 22.
            # The loop `for n in range(10, 23)` handles this.
    
    # Verify best result
    if best_n >= 6 and best_edges:
        adj = [0] * best_n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Final verification
        is_valid = True
        for k in range(6, best_n + 1):
            for subset in itertools.combinations(range(best_n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    d = (adj[v] & mask).bit_count()
                    degs.append(d)
                if all(d == degs[0] for d in degs):
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if not is_valid:
            best_n = 5
            best_edges = []
            
    return {'n': best_n, 'edges': best_edges}
