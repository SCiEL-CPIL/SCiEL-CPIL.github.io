import time
import numpy as np
from collections import deque

def _build_luw(d, q):
    """
    Build the Lazebnik-Ustimenko-Woldar graph CD(d, q).
    n = 2 * q^d, degree = q, girth >= 2d+4 (for d>=3).
    Vertices: (side, x) where side in {0,1}, x in GF(q)^d.
    """
    q_int = int(q)
    
    # Rule for computing y[c-1] from previous y values and x
    # For c in 2..d:
    #   y[c-1] = x[c-1] + y[a-1] * x[b-1]  (side 0)
    #   y[c-1] = x[c-1] - x[a-1] * y[b-1]  (side 1)
    # where (a, b) = rule[c]
    
    rule = {}
    # Standard LUW rules for girth maximization
    # For d=3: rule[2]=(1,1), rule[3]=(2,1)
    # For d=4: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2)
    # For d=5: rule[2]=(1,1), rule[3]=(2,1), rule[4]=(1,2), rule[5]=(1,3)
    # General pattern:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # c=5: (1,3)
    # c=6: (2,3)? or (1,4)?
    # Let's use the pattern from literature:
    # The rule is typically y_j = x_j + y_{j-1} * x_{j-2} for odd j, etc.
    # A common valid rule set for CD(d,q):
    # a(c) = c-1 if c is even, c-2 if c is odd?
    # b(c) = 1 if c is even, c-2 if c is odd?
    # Let's stick to a known valid sequence for small d.
    
    # Known valid rules for d up to 5:
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    if d >= 5: rule[5] = (1, 3)
    if d >= 6: rule[6] = (2, 3)
    if d >= 7: rule[7] = (1, 4)
    
    # If d is larger, extend with a pattern that maintains girth.
    # A simple pattern that works for many cases:
    # For c >= 2:
    # a = c - 1
    # b = 1 if c % 2 == 0 else c - 2
    # Let's verify this pattern:
    # c=2: a=1, b=1 (even c -> b=1). Matches (1,1).
    # c=3: a=2, b=1 (odd c -> b=c-2=1). Matches (2,1).
    # c=4: a=3, b=1 (even c -> b=1). Wait, previous was (1,2).
    # The pattern a=c-1, b=1 for even c gives (3,1) for c=4.
    # The previous explicit rule was (1,2).
    # Different rules give different girths. The explicit ones are known to be good.
    # For c > 7, let's use the a=c-1, b=1 pattern as a fallback, or just stop at d=7.
    # Since we only need d=3 for the best result, and maybe d=4,5 as fallback,
    # the explicit rules up to 5 are sufficient.
    
    for c in range(6, d + 1):
        # Use a generic pattern
        if c % 2 == 0:
            rule[c] = (c - 1, 1)
        else:
            rule[c] = (c - 1, c - 2)
            
    # Precompute neighbor relations
    # Vertices are indexed 0 to 2*q^d - 1.
    # Index = side * q^d + sum(x_i * q^(d-1-i))
    
    n = 2 * (q_int ** d)
    adj = [[] for _ in range(n)]
    edges = []
    
    # Helper to convert tuple to index
    def tuple_to_idx(side, x):
        val = 0
        for i in range(d):
            val = val * q_int + x[i]
        return side * (q_int ** d) + val
    
    # Helper to convert index to tuple
    def idx_to_tuple(idx):
        side = 0 if idx < q_int ** d else 1
        rem = idx % (q_int ** d)
        x = [0] * d
        for i in range(d - 1, -1, -1):
            x[i] = rem % q_int
            rem //= q_int
        return side, tuple(x)
    
    # We only need to generate edges from side 0 to side 1 to avoid duplicates
    # For each x in GF(q)^d:
    #   For each y_0 in GF(q):
    #     Compute y_1...y_{d-1}
    #     If side 0: y[c-1] = (x[c-1] + y[a-1] * x[b-1]) % q
    #     This defines a neighbor (1, y) for (0, x).
    #     Is it a bijection? Yes, for each x and y_0, y is uniquely determined.
    #     So each (0, x) has q neighbors in side 1.
    #     Each (1, y) has q neighbors in side 0.
    
    # Let's iterate over all x in side 0
    for x_val in range(q_int ** d):
        x = idx_to_tuple(x_val)[1]
        for y0 in range(q_int):
            y = [0] * d
            y[0] = y0
            for c in range(2, d + 1):
                a, b = rule[c]
                # y[c-1] = x[c-1] + y[a-1] * x[b-1]
                y[c-1] = (x[c-1] + y[a-1] * x[b-1]) % q_int
            
            y_idx = 1 * (q_int ** d) + sum(y[i] * (q_int ** (d - 1 - i)) for i in range(d))
            x_idx = x_val
            
            if x_idx < y_idx:
                adj[x_idx].append(y_idx)
                adj[y_idx].append(x_idx)
                edges.append((x_idx, y_idx))
            else:
                adj[x_idx].append(y_idx)
                adj[y_idx].append(x_idx)
                edges.append((y_idx, x_idx))
                
    return n, edges, adj

def _girth_check(adj, n, g):
    """
    Check if the graph has girth at least g.
    Returns True if girth >= g, False otherwise.
    """
    h = (g - 1) // 2
    # For each vertex, do a BFS up to depth h.
    # If we find a cycle of length < g, return False.
    # A cycle of length L corresponds to a path of length L-1 from a vertex back to itself?
    # No, simpler: if there is a vertex v such that dist(s, v) + dist(s, v) < g? No.
    # If there is a cycle of length L < g, then there exists a vertex s on the cycle
    # such that two distinct paths of length L/2 (if L even) or (L-1)/2 and (L+1)/2 (if L odd) meet?
    
    # Standard check:
    # For each start s:
    #   BFS from s.
    #   If we encounter a vertex v that has already been visited, and v is not the parent of the current node,
    #   we found a cycle. The length of the cycle is depth[u] + depth[v] + 1.
    #   We must check if this length is < g.
    
    for s in range(n):
        depth = [-1] * n
        parent = [-1] * n
        depth[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if depth[u] >= h:
                continue
            for v in adj[u]:
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        # Found a cycle
                        cycle_len = depth[u] + depth[v] + 1
                        if cycle_len < g:
                            return False
    return True

def construct(k, g):
    deadline = time.time() + 115.0
    best_n = float('inf')
    best_edges = None
    best_adj = None
    
    q = int(k)
    
    # Try d=3 first (n=250 for k=5)
    # Girth of CD(3, q) is at least 10. So it satisfies g=9.
    # Try d=4 (n=1250)
    # Girth of CD(4, q) is at least 12.
    
    for d in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges, adj = _build_luw(d, q)
        except Exception:
            continue
            
        # Check validity
        # 1. Regularity
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break
        if not valid:
            continue
            
        # 2. Girth
        if _girth_check(adj, n, g):
            if n < best_n:
                best_n = n
                best_edges = edges
                best_adj = adj
            break # Smaller d is better, so stop if valid
            
    if best_edges is None:
        # Fallback: return a trivial invalid result or raise
        # This should not happen for k=5, g=9
        raise ValueError("No valid graph found")
        
    return best_n, best_edges
