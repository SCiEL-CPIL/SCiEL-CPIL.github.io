import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses a greedy randomized construction for small parameters (like 4,10).
    Falls back to Lazebnik-Ustimenko-Woldar construction if greedy fails or for larger parameters.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    # Moore bound calculation for even girth
    # n >= 2*k * sum_{i=0}^{g/2 - 2} (k-1)^i
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * k * sum((k - 1) ** i for i in range(r - 1))
    else:
        # Odd girth bound is different, but task is g=10
        r = (g - 1) // 2
        moore_bound = 1 + k * sum((k - 1) ** i for i in range(r - 1))

    # We aim for the Moore bound or slightly above
    target_n = moore_bound
    
    # Ensure target_n is at least 2*k
    if target_n < 2 * k:
        target_n = 2 * k

    best_n = None
    best_edges = None

    # Try greedy construction starting from Moore bound
    # Step size 2 to maintain regularity possibilities (though k=4 allows any n >= 2k? No, kn must be even)
    # For k=4, 4n is always even.
    
    # We try a few values around the Moore bound
    # If Moore bound is 320, try 320, 322, 324...
    
    # Optimization: If k, g are not 4, 10, greedy might be too slow or fail.
    # LUW is safer for general cases, but greedy is better for small cages.
    
    use_greedy = (k == 4 and g == 10) or (k <= 6 and g <= 12)
    
    if use_greedy:
        for n_try in range(target_n, target_n + 100, 2):
            if time.time() > deadline:
                break
            
            success, edges = _try_greedy_construction(n_try, k, g)
            if success:
                best_n = n_try
                best_edges = edges
                break
        
        # If greedy failed for small n, try a bit larger range with more restarts?
        # The loop above tries different n.
        # If it fails for all, we fall back.
        
    if best_edges is None:
        # Fallback to LUW
        best_n, best_edges = _luw_construct(k, g)
        
    return best_n, best_edges

def _try_greedy_construction(n, k, g):
    """
    Attempts to construct a k-regular graph of n vertices with girth >= g
    using a greedy edge addition strategy.
    """
    # Random seed for reproducibility in a single run? 
    # Using global random is fine.
    
    adj = [set() for _ in range(n)]
    degrees = [0] * n
    total_edges = n * k // 2
    
    # To avoid getting stuck, we can use a heuristic:
    # Always pick u with min degree.
    # Pick v with min degree among valid candidates.
    
    for _ in range(total_edges):
        # Find u with minimum degree
        min_deg_u = k
        candidates_u = []
        for i in range(n):
            if degrees[i] < k:
                if degrees[i] < min_deg_u:
                    min_deg_u = degrees[i]
                    candidates_u = [i]
                elif degrees[i] == min_deg_u:
                    candidates_u.append(i)
        
        if not candidates_u:
            # Should not happen
            return False, []
            
        u = random.choice(candidates_u)
        
        # Determine forbidden vertices for u
        # Forbidden if:
        # 1. v == u
        # 2. v in adj[u]
        # 3. Path from u to v of length < g-1 exists
        
        # BFS from u up to depth g-2
        max_depth = g - 2
        visited = [False] * n
        queue = deque()
        queue.append((u, 0))
        visited[u] = True
        
        while queue:
            node, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append((nb, dist + 1))
        
        valid_v = []
        min_deg_v = k
        for v in range(n):
            if v == u:
                continue
            if degrees[v] >= k:
                continue
            if v in adj[u]:
                continue
            if visited[v]:
                continue
            
            valid_v.append(v)
            if degrees[v] < min_deg_v:
                min_deg_v = degrees[v]
        
        if not valid_v:
            return False, []
        
        # Filter valid_v to only those with min degree
        candidates_v = [v for v in valid_v if degrees[v] == min_deg_v]
        v = random.choice(candidates_v)
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
        
    # Verify degrees
    if all(d == k for d in degrees):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return True, edges
    else:
        return False, []

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction fallback.
    """
    q = k
    # d is the parameter for LUW. 
    # Girth is d+5 (if d odd) or d+4 (if d even)?
    # Original code: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
