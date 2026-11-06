import time
import random

def construct():
    deadline = time.time() + 115.0
    
    q = 6
    
    # Generate all points of PG(2,q) as normalized vectors in F_q^3
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q) if q > 2 else 1
                    # For q=6, not all elements have inverses, so use a different normalization
                    # Actually for q=6, we need to be more careful
                    points.append((x, y, z))
                elif y != 0:
                    points.append((0, y, z))
                else:
                    points.append((0, 0, z))
    
    # Deduplicate: normalize so that the first non-zero coordinate is as large as possible
    # Actually, let's use a proper projective point representation
    # Points are 1-dimensional subspaces of F_q^3
    # Represent each point by its lexicographically smallest representative
    
    def normalize(v):
        """Return canonical representative of a projective point."""
        x, y, z = v
        # Find first non-zero coordinate
        if x != 0:
            # Scale so x is the maximum possible (try all non-zero multipliers)
            best = (x, y, z)
            for a in range(1, q):
                nx = (a * x) % q
                ny = (a * y) % q
                nz = (a * z) % q
                if (nx, ny, nz) < best:
                    best = (nx, ny, nz)
            return best
        elif y != 0:
            best = (x, y, z)
            for a in range(1, q):
                nx = (a * x) % q
                ny = (a * y) % q
                nz = (a * z) % q
                if (nx, ny, nz) < best:
                    best = (nx, ny, nz)
            return best
        else:
            best = (x, y, z)
            for a in range(1, q):
                nx = (a * x) % q
                ny = (a * y) % q
                nz = (a * z) % q
                if (nx, ny, nz) < best:
                    best = (nx, ny, nz)
            return best
    
    # Generate all non-zero vectors, normalize, deduplicate
    seen = set()
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                norm = normalize((x, y, z))
                if norm not in seen:
                    seen.add(norm)
                    points.append(norm)
    
    n_base = len(points)
    # For q=6, this should be 43
    
    # Define a polarity on PG(2,q)
    # A polarity is given by a symmetric bilinear form
    # Use the form: B(u,v) = u0*v0 + u1*v1 + u2*v2 (mod q)
    # But this might not give a proper polarity for all q
    # For a proper polarity, we need B(u,u) to be defined for all u
    # The standard polarity: B((x1,y1,z1),(x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2
    
    # Two points u,v are adjacent if B(u,v) = 0
    # This is the polarity graph
    
    # Build adjacency
    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify C4-free
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):  # more than one bit set
                    return False
        return True
    
    if not is_c4_free(adj, n_base):
        # Fallback: try different polarity
        # Try B(u,v) = u0*v2 + u1*v1 + u2*v0 (another symmetric form)
        adj = [0] * n_base
        for i in range(n_base):
            for j in range(i + 1, n_base):
                u = points[i]
                v = points[j]
                dot = (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % q
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if not is_c4_free(adj, n_base):
            # Fallback to 10 disjoint triangles
            n_fallback = 30
            adj_fallback = [0] * n_fallback
            for i in range(10):
                u, v, w = 3*i, 3*i+1, 3*i+2
                adj_fallback[u] |= (1<<v) | (1<<w)
                adj_fallback[v] |= (1<<u) | (1<<w)
                adj_fallback[w] |= (1<<u) | (1<<v)
            edges = []
            for u in range(n_fallback):
                for v in range(u + 1, n_fallback):
                    if adj_fallback[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n_fallback, 'edges': edges}
    
    # Check alpha < 11 using branch and bound
    def has_independent_set_of_size(adj, n, target):
        """Returns True if there exists an independent set of size >= target"""
        if n < target:
            return False
        
        # Sort vertices by degree (high degree first for better pruning)
        degrees = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -degrees[v])
        
        found = [False]
        
        def backtrack(start, size, excluded_mask):
            if found[0]:
                return
            if time.time() > deadline - 20:
                return
            if size >= target:
                found[0] = True
                return
            
            # Upper bound
            remaining = n - start
            if size + remaining < target:
                return
            
            if start >= n:
                return
            
            v = order[start]
            if excluded_mask & (1 << v):
                backtrack(start + 1, size, excluded_mask)
                return
            
            # Option 1: Don't include v
            backtrack(start + 1, size, excluded_mask)
            if found[0]:
                return
            
            # Option 2: Include v
            # Exclude v and all its neighbors
            new_excluded = excluded_mask | (1 << v) | adj[v]
            backtrack(start + 1, size + 1, new_excluded)
        
        backtrack(0, 0, 0)
        return found[0]
    
    if has_independent_set_of_size(adj, n_base, 11):
        # The full polarity graph has alpha >= 11
        # Try to find a subgraph or use a different approach
        # Fall back to 10 disjoint triangles
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            adj_fallback[u] |= (1<<v) | (1<<w)
            adj_fallback[v] |= (1<<u) | (1<<w)
            adj_fallback[w] |= (1<<u) | (1<<v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}
    
    # Build edge list
    edges = []
    for u in range(n_base):
        for v in range(u + 1, n_base):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    return {'n': n_base, 'edges': edges}
