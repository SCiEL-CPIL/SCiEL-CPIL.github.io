import time
import random
import numpy as np

def _build_adj_masks(colors, n):
    """Build adjacency bitmasks for both colors."""
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    return adj1, adj0

def _count_k222_in_adj(adj, n):
    """Count number of K2,2,2 subgraphs in a graph."""
    count = 0
    # Precompute lists of vertices for masks to speed up iteration
    # But since n <= 31, iterating bits is fast.
    
    for u in range(n):
        # u is in part 1
        # v must be non-adjacent to u
        # Let's iterate v > u to avoid double counting? 
        # K2,2,2 has 3 parts of size 2. 
        # A subgraph is defined by 6 vertices.
        # We can count each subgraph 3! * 2! * 2! * 2! times? No.
        # Let's just count distinct sets of 6 vertices that form K2,2,2.
        # The structure is: Parts P1, P2, P3.
        # Edges exist between any vertex in Pi and Pj for i!=j.
        # No edges within Pi.
        
        # We can iterate over pairs (u,v) in P1.
        # Then find common neighbors C.
        # Then find pairs (a,b) in C that are non-adjacent (to form P2).
        # Then find common neighbors of a,b in C (excluding u,v,a,b) that are non-adjacent to each other (to form P3).
        
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue # u,v must be in same part, so non-adjacent
            
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            
            # Extract vertices in C
            verts_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                verts_c.append(lsb.bit_length() - 1)
                mask ^= lsb
                
            m = len(verts_c)
            # Find pairs (a,b) in C that are non-adjacent
            # These will form P2
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj[a] & (1 << b):
                        continue # a,b must be in same part P2, so non-adjacent
                    
                    # Now we need P3.
                    # Vertices in P3 must be adjacent to u, v, a, b.
                    # So they must be in C (adj to u,v) AND adj[a] AND adj[b].
                    # And they must be non-adjacent to each other.
                    
                    common = C & adj[a] & adj[b]
                    # Exclude u, v, a, b from common? 
                    # u,v are not in C (C is neighbors of u,v). 
                    # a,b are in C. Are a,b adjacent to each other? No.
                    # So a,b are in common? 
                    # adj[a] contains neighbors of a. b is not a neighbor of a.
                    # So b is NOT in adj[a]. Thus b is not in common.
                    # Similarly a is not in common.
                    # So common contains only candidates for P3.
                    
                    if common.bit_count() < 2:
                        continue
                    
                    # Find pairs (c,d) in common that are non-adjacent
                    verts_common = []
                    mask2 = common
                    while mask2:
                        lsb = mask2 & -mask2
                        verts_common.append(lsb.bit_length() - 1)
                        mask2 ^= lsb
                        
                    k = len(verts_common)
                    for x in range(k):
                        c = verts_common[x]
                        for y in range(x + 1, k):
                            d = verts_common[y]
                            if not (adj[c] & (1 << d)):
                                # Found a K2,2,2
                                count += 1
    return count

def _is_valid(adj1, adj0, n):
    return _count_k222_in_adj(adj1, n) == 0 and _count_k222_in_adj(adj0, n) == 0

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _local_search_sa(colors, n, time_limit=50):
    """Simulated Annealing style local search to minimize K2,2,2 violations."""
    start_time = time.time()
    # Copy colors to avoid modifying original
    c = colors[:]
    
    # Initial state
    adj1, adj0 = _build_adj_masks(c, n)
    curr_viol = _count_k222_in_adj(adj1, n) + _count_k222_in_adj(adj0, n)
    
    if curr_viol == 0:
        return c, True
    
    best_c = c[:]
    best_viol = curr_viol
    
    # Precompute edge index
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if curr_viol == 0:
            return c, True
            
        # Pick a random edge to flip
        # Bias towards edges that might be in violations? 
        # For simplicity, pick random edge first.
        # If we want to be smarter, we could pick an edge involved in a violation.
        # But counting violations gives us the number, not the edges.
        # Let's stick to random flip for now, SA handles the landscape.
        
        idx_flip = random.randint(0, len(c) - 1)
        old_color = c[idx_flip]
        c[idx_flip] = 1 - old_color
        
        # Update adjacency masks incrementally? 
        # Rebuilding is O(n^2) ~ 800 ops. 
        # Counting is O(n^6) worst case but pruned. 
        # Let's rebuild to be safe and simple, it's fast enough for n=29.
        
        adj1, adj0 = _build_adj_masks(c, n)
        new_viol = _count_k222_in_adj(adj1, n) + _count_k222_in_adj(adj0, n)
        
        delta = new_viol - curr_viol
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            curr_viol = new_viol
            if curr_viol < best_viol:
                best_viol = curr_viol
                best_c = c[:]
                if curr_viol == 0:
                    return c, True
        else:
            # Reject
            c[idx_flip] = old_color
            
        T *= cooling
        if T < 0.01:
            # Restart with higher T if stuck?
            # Or just continue with low T (greedy)
            pass
            
    return best_c, best_viol == 0

def construct():
    deadline = time.time() + 140 # Safety margin
    
    # Strategy:
    # 1. Try to find n=29.
    # 2. If fails, ensure we have a good n=28.
    
    # Attempt n=29
    n_target = 29
    found = False
    
    # Try Paley(29)
    p_target = 29
    colors = _paley(n_target, p_target)
    if colors:
        adj1, adj0 = _build_adj_masks(colors, n_target)
        if _is_valid(adj1, adj0, n_target):
            return {'n': n_target, 'colors': colors}
        
        remaining = deadline - time.time()
        if remaining > 10:
            c_res, is_valid = _local_search_sa(colors, n_target, time_limit=int(remaining - 5))
            if is_valid:
                return {'n': n_target, 'colors': c_res}
        
    # Try other Paley primes for n=29
    primes = [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    for p in primes:
        if time.time() > deadline - 20:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        adj1, adj0 = _build_adj_masks(colors, 29)
        if _is_valid(adj1, adj0, 29):
            return {'n': 29, 'colors': colors}
        
        remaining = deadline - time.time()
        if remaining > 15:
            c_res, is_valid = _local_search_sa(colors, 29, time_limit=int(remaining - 10))
            if is_valid:
                return {'n': 29, 'colors': c_res}
                
    # Try random starts for n=29
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 15:
            c_res, is_valid = _local_search_sa(colors, 29, time_limit=int(remaining - 10))
            if is_valid:
                return {'n': 29, 'colors': c_res}

    # Fallback to n=28
    n_target = 28
    # Try Paley(29) restricted to 28? No, Paley(28) doesn't exist directly, use Paley(29) and drop a vertex?
    # Or use Paley with p=29, take first 28 vertices.
    colors = _paley(28, 29)
    if colors:
        adj1, adj0 = _build_adj_masks(colors, 28)
        if _is_valid(adj1, adj0, 28):
            return {'n': 28, 'colors': colors}
        
        remaining = deadline - time.time()
        if remaining > 5:
            c_res, is_valid = _local_search_sa(colors, 28, time_limit=int(remaining - 2))
            if is_valid:
                return {'n': 28, 'colors': c_res}
                
    # Random start for n=28
    colors = [random.randint(0, 1) for _ in range(28 * 27 // 2)]
    remaining = deadline - time.time()
    if remaining > 5:
        c_res, is_valid = _local_search_sa(colors, 28, time_limit=int(remaining - 2))
        if is_valid:
            return {'n': 28, 'colors': c_res}
            
    # Absolute fallback
    return {'n': 1, 'colors': []}
