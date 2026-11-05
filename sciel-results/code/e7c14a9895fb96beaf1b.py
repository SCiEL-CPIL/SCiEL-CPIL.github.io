import time
import itertools
import random

def _polarity_graph(q):
    """
    Generates the polarity graph of the projective plane of order q.
    Vertices are points of PG(2, q).
    Two points are adjacent if one lies on the polar of the other.
    This graph is C4-free.
    """
    if q < 2:
        return 0, []
    
    # Points in PG(2, q) can be represented as (x, y, 1) and (x, 0, 0) etc.
    # Total points = q^2 + q + 1
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 0, 0))
    points.append((1, 0, 0))
    
    n = len(points)
    
    # Define the polarity: (x, y, z) -> (y, -x, z) ? 
    # Standard polarity for Desarguesian plane:
    # A point P=(x,y,z) is polar to line xX + yY + zZ = 0? No.
    # Let's use the standard orthogonal polarity:
    # P=(x,y,z) is adjacent to Q=(u,v,w) if xu + yv + zw = 0.
    # This is symmetric and irreflexive if q is even? 
    # If q is odd, P is adjacent to P if xx+yy+zz=0. 
    # For q=5, 1^2+0^2+0^2 = 1 != 0. 0^2+1^2+0^2=1!=0.
    # We need to check if self-loops are formed.
    # If self-loops exist, we remove them.
    
    edges = []
    adj = [[0]*n for _ in range(n)]
    
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i+1, n):
            x2, y2, z2 = points[j]
            # Check if x1*x2 + y1*y2 + z1*z2 == 0 mod q
            if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                # Check if it's a self-loop (i==j is handled by i<j, but check if point is on its own polar)
                # If i != j, it's an edge.
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
                
    return n, edges

def _has_independent_set(adj, n, k, time_limit=5.0):
    """
    Returns True if an independent set of size k exists.
    Uses a simple branch and bound.
    """
    start_time = time.time()
    
    # Sort vertices by degree (heuristic)
    degrees = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    # Remap indices
    remap = [0]*n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
        
    new_adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                new_adj[remap[i]][remap[j]] = 1
                
    # Use bitsets for speed
    adj_bits = [0]*n
    for i in range(n):
        bits = 0
        for j in range(n):
            if new_adj[i][j]:
                bits |= (1 << j)
        adj_bits[i] = bits
        
    # We want to find a set of k vertices with no edges between them.
    # This is equivalent to finding a clique of size k in the complement graph.
    # Complement graph edge i-j exists if NOT adj[i][j] and i!=j.
    
    comp_bits = [0]*n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = (all_mask ^ adj_bits[i]) & ~(1 << i)
        
    found = False
    
    def backtrack(start, current_set, size):
        nonlocal found
        if time.time() - start_time > time_limit:
            return
        if size == k:
            found = True
            return
            
        # Pruning: if remaining vertices + current size < k, return
        if n - start + size < k:
            return
            
        # Heuristic: pick a vertex from 'start'
        for i in range(start, n):
            if found:
                return
            # If we pick i, we can only pick vertices j > i that are non-adjacent to i in original graph
            # i.e., adjacent to i in complement graph.
            # And we must ensure they are also non-adjacent to all in current_set.
            
            # Check if i is compatible with current_set
            compatible = True
            for v in current_set:
                if not (comp_bits[v] & (1 << i)):
                    compatible = False
                    break
            if not compatible:
                continue
                
            # New candidate set for next recursion
            # Must be vertices j > i that are compatible with i and current_set
            # Intersection of comp_bits[i] and intersection of comp_bits[v] for v in current_set
            mask = comp_bits[i]
            for v in current_set:
                mask &= comp_bits[v]
                
            # Only consider vertices > i
            mask &= ~((1 << (i+1)) - 1)
            
            if mask == 0:
                continue
                
            # Recurse
            # We need to iterate through bits in mask
            # Simple loop
            m = mask
            while m:
                if found:
                    return
                lsb = m & -m
                j = lsb.bit_length() - 1
                m ^= lsb
                
                # Recurse with j added
                # Note: we are adding j to current_set
                backtrack(j+1, current_set + [j], size + 1)
                
    backtrack(0, [], 0)
    return found

def construct():
    deadline = time.time() + 110  # Leave 10s margin
    
    best_n = 0
    best_edges = []
    
    # Try q=5 (n=31)
    if time.time() < deadline - 20:
        try:
            n, edges = _polarity_graph(5)
            adj = [[0]*n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = 1
                adj[v][u] = 1
                
            # Check C4 (should be free by construction, but verify)
            # Check Independent Set
            has_is = _has_independent_set(adj, n, 11, time_limit=10.0)
            if not has_is:
                if n > best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
            
    # Try q=7 (n=57)
    if time.time() < deadline - 30:
        try:
            n, edges = _polarity_graph(7)
            adj = [[0]*n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = 1
                adj[v][u] = 1
                
            # Check Independent Set
            # This might be slow. Limit time.
            has_is = _has_independent_set(adj, n, 11, time_limit=20.0)
            if not has_is:
                if n > best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
            
    if best_n == 0:
        # Fallback: empty graph? No, n>=1.
        # Return a single vertex
        return {'n': 1, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
