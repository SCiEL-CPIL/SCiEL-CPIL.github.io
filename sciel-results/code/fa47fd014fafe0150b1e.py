import time
import random

def _has_k222(adj):
    n = len(adj)
    # adj[i] is a bitmask of neighbors
    # Check for K2,2,2
    # Partitions A={u,v}, B={w,x}, C={y,z}
    # u,v not connected. w,x not connected. y,z not connected.
    # All cross edges present.
    
    # Precompute non-neighbor masks for faster lookup
    # non_adj[i] = mask of vertices NOT connected to i (excluding i itself)
    full_mask = (1 << n) - 1
    non_adj = []
    for i in range(n):
        non_adj.append(full_mask ^ adj[i] ^ (1 << i))
    
    # Iterate over pairs (u, v) not connected
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                continue # u and v connected, cannot be same part
            
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            
            # We need to find two disjoint non-edges (w,x) and (y,z) in common
            # such that all edges between {w,x} and {y,z} are present.
            
            # Get list of vertices in common
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            
            # Iterate over pairs (w, x) in verts
            # w and x must NOT be connected
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i+1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue # w and x connected, cannot be same part
                    
                    # Now we have part A={u,v}, part B={w,x}
                    # We need part C={y,z} such that:
                    # y,z not connected
                    # y,z connected to u,v,w,x
                    # y,z are in common (so connected to u,v)
                    # y,z must be connected to w and x
                    
                    # Candidates for C are in common & adj[w] & adj[x]
                    cand = common & adj[w] & adj[x]
                    
                    # Get list of candidates
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                        
                    # Check if cverts contains a non-edge pair
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k+1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    def build_adj(n, colors):
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

    def check_coloring(n, colors):
        adj0, adj1 = build_adj(n, colors)
        return not (_has_k222(adj0) or _has_k222(adj1))

    def find_k222(adj):
        """Find a K2,2,2 in adj, return (u,v,w,x,y,z) or None."""
        n = len(adj)
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

    def local_search(n, colors, max_time=100):
        """Local search: find K222, flip a random edge in it."""
        start = time.time()
        deadline = start + max_time
        colors = list(colors)
        n_edges = n * (n - 1) // 2
        
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        while time.time() < deadline:
            adj0, adj1 = build_adj(n, colors)
            k222_0 = find_k222(adj0)
            k222_1 = find_k222(adj1)
            
            if k222_0 is None and k222_1 is None:
                return colors  # Valid!
            
            # Pick which K222 to break
            if k222_0 is not None and (k222_1 is None or random.random() < 0.5):
                u, v, w, x, y, z = k222_0
                color = 0
            else:
                u, v, w, x, y, z = k222_1
                color = 1
            
            # The K222 has 6 vertices with 12 cross-edges
            # Flip a random cross-edge
            verts = [u, v, w, x, y, z]
            parts = [[u, v], [w, x], [y, z]]
            # All cross-edges between different parts
            cross_edges = []
            for pi in range(3):
                for pj in range(pi+1, 3):
                    for a in parts[pi]:
                        for b in parts[pj]:
                            cross_edges.append((a, b))
            
            ei, ej = random.choice(cross_edges)
            idx = edge_idx(ei, ej)
            colors[idx] = 1 - colors[idx]
        
        return colors  # Return best effort

    def paley(n):
        # Only works if n is prime and n % 4 == 1
        # Check primality and mod
        if n < 2: return None
        if n % 4 != 1: return None
        # Simple primality check
        if n % 2 == 0: return None
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return None
            
        squares = {x*x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                if (j-i) % n in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try known good sizes first, descending from 29
    for n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
            
        # Try Paley if possible
        if n % 4 == 1:
            colors = paley(n)
            if colors and check_coloring(n, colors):
                return {'n': n, 'colors': colors}
        
        # If Paley failed or not applicable, try local search for this n
        if n >= 20:
            remaining = deadline - time.time()
            if remaining > 5:
                # Start from Paley if available, else random
                if n % 4 == 1:
                    init_colors = paley(n)
                else:
                    init_colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
                
                if init_colors:
                    result = local_search(n, init_colors, max_time=min(remaining - 3, 30))
                    if check_coloring(n, result):
                        return {'n': n, 'colors': result}
                
                # Also try random restarts
                for _ in range(3):
                    if time.time() > deadline - 3:
                        break
                    rand_colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
                    result = local_search(n, rand_colors, max_time=min(deadline - time.time() - 2, 20))
                    if check_coloring(n, result):
                        return {'n': n, 'colors': result}

    # Fallback to seed
    return {'n': 17, 'colors': paley(17)}
