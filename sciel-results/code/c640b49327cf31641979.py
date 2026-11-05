import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n from 47 down to 39
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        target_min_deg = n - 39
        
        # For n=39, target is 0. Any graph works.
        # For n=40, target is 1.
        # ...
        
        found = False
        best_min_deg = -1
        best_adj = None
        
        # Determine restart budget
        # If n is close to 47, we have less time per restart effectively? No, restart is fast.
        # Just run as many as possible until deadline.
        
        while time.time() < deadline:
            adj = [0] * n
            
            # Greedy construction
            vertices = list(range(n))
            random.shuffle(vertices)
            
            for u in vertices:
                others = [v for v in range(n) if v != u]
                random.shuffle(others)
                
                for v in others:
                    if u > v: continue
                    
                    common = adj[u] & adj[v]
                    if common.bit_count() < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            min_deg = min((adj[u].bit_count() for u in range(n)), default=0)
            
            if min_deg > best_min_deg:
                best_min_deg = min_deg
                best_adj = adj[:]
                
            if min_deg >= target_min_deg:
                # Extract edges
                edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
            
            # If we are running out of time, break to try next n?
            # No, if we fail n=47, we should try n=46.
            # But we should use the remaining time for n=46.
            # The while loop will continue until deadline.
            # But if we exit the while loop (time out), we go to next n.
            # This is correct.

        # If loop finished due to timeout, we move to next n.
        # If loop finished because... well, it only breaks on timeout or return.
        # So if we didn't return, we timed out for this n.
        # Move to next n.
        
    return {'n': 39, 'edges': []}
