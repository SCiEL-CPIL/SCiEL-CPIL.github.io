import time
import random

def _has_k222(adj):
    """Check if bitmask-adjacency graph contains K2,2,2 (complete tripartite, parts of size 2)."""
    n = len(adj)
    if n < 6:
        return False
    full_mask = (1 << n) - 1
    
    # Iterate over non-edges (u,v) as first part
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                continue  # u,v connected, can't be in same part
            
            # Common neighbors of u and v (must be connected to both)
            common = adj[u] & adj[v]
            
            # Get list of vertices in common
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            
            # Try each non-edge (w,x) in verts as second part
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i+1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue  # w,x connected, can't be in same part
                    
                    # Candidates for third part: connected to u,v (in common) and to w,x
                    cand = common & adj[w] & adj[x]
                    
                    # Check if cand contains a non-edge pair (y,z)
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                    
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k+1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return True
    return False


def _find_k222(adj):
    """Find a K2,2,2 in adj, return (u,v,w,x,y,z) or None."""
    n = len(adj)
    if n < 6:
        return None
    
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                continue
            
            common = adj[u] & adj[v]
            
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i+1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue
                    
                    cand = common & adj[w] & adj[x]
                    
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                    
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k+1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return (u, v, w, x, y, z)
    return None


def _build_adj_bitmask(n, colors):
    """Build bitmask adjacency lists for both color classes."""
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    return adj0, adj1


def _check_valid(n, colors):
    """Check if coloring is valid (no K2,2,2 in either color)."""
    if len(colors) != n*(n-1)//2:
        return False
    adj0, adj1 = _build_adj_bitmask(n, colors)
    return not (_has_k222(adj0) or _has_k222(adj1))


def _paley_colors(n, p):
    """Paley coloring: color 0 if (j-i) mod p is a quadratic residue mod p."""
    if p < n:
        return None
    squares = {x*x % p for x in range(1, p)}
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(0 if (j-i) % p in squares else 1)
    return colors


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    def _edge_idx(i, j, n):
        """Compute lexicographic edge index for pair (i,j) with i<j."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def _local_search(n, colors, max_time=30.0):
        """Local search: find K2,2,2 in either color, flip a random cross-edge in it."""
        t_start = time.time()
        t_dead = t_start + max_time
        colors = list(colors)
        n_edges = n * (n - 1) // 2
        
        while time.time() < t_dead:
            adj0, adj1 = _build_adj_bitmask(n, colors)
            k222_0 = _find_k222(adj0)
            k222_1 = _find_k222(adj1)
            
            if k222_0 is None and k222_1 is None:
                return colors
            
            # Choose which violation to fix
            if k222_0 is not None and (k222_1 is None or random.random() < 0.5):
                u, v, w, x, y, z = k222_0
            else:
                u, v, w, x, y, z = k222_1
            
            # Cross-edges between the three pairs
            parts = [[u, v], [w, x], [y, z]]
            cross_edges = []
            for pi in range(3):
                for pj in range(pi+1, 3):
                    for a in parts[pi]:
                        for b in parts[pj]:
                            cross_edges.append((a, b))
            
            ei, ej = random.choice(cross_edges)
            idx = _edge_idx(ei, ej, n)
            colors[idx] = 1 - colors[idx]
        
        return colors
    
    # Strategy 1: Try Paley graph of order 29 (known to work)
    # The Paley graph on 29 vertices is strongly regular (29,14,6,7)
    # and both it and its complement are K2,2,2-free.
    if time.time() < deadline - 10:
        colors29 = _paley_colors(29, 29)
        if colors29 is not None and _check_valid(29, colors29):
            return {'n': 29, 'colors': colors29}
    
    # Strategy 2: Try other Paley constructions (primes p where n <= p)
    primes_to_try = [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes_to_try:
        if time.time() > deadline - 15:
            break
        # Try n = p (full Paley) if p <= 31
        if p <= 31:
            colors = _paley_colors(p, p)
            if colors is not None and _check_valid(p, colors):
                if p >= 29:
                    return {'n': p, 'colors': colors}
        # Try n = 29 with larger prime p (subgraph of Paley(p))
        if p > 29:
            colors = _paley_colors(29, p)
            if colors is not None and _check_valid(29, colors):
                return {'n': 29, 'colors': colors}
    
    # Strategy 3: Local search from Paley(29) or random init for n=29, 28, ...
    for n_target in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time() - 5
        
        # Try from Paley init
        init_colors = None
        if n_target % 4 == 1:
            init_colors = _paley_colors(n_target, n_target)
        if init_colors is None:
            # Try with next prime
            for p in [n_target+1, n_target+2, n_target+4, n_target+6]:
                if p >= n_target:
                    c = _paley_colors(n_target, p)
                    if c is not None:
                        init_colors = c
                        break
        
        if init_colors is not None:
            result = _local_search(n_target, init_colors, max_time=min(remaining * 0.4, 25.0))
            if _check_valid(n_target, result):
                return {'n': n_target, 'colors': result}
        
        # Try random restarts
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            rand_colors = [random.randint(0, 1) for _ in range(n_target*(n_target-1)//2)]
            result = _local_search(n_target, rand_colors, max_time=min(remaining * 0.15, 15.0))
            if _check_valid(n_target, result):
                return {'n': n_target, 'colors': result}
    
    # Strategy 4: Fallback - return best we can verify quickly
    # Try n=17 Paley as guaranteed fallback
    colors17 = _paley_colors(17, 17)
    if colors17 is not None and _check_valid(17, colors17):
        return {'n': 17, 'colors': colors17}
    
    # Absolute fallback
    return {'n': 17, 'colors': _paley_colors(17, 17)}
