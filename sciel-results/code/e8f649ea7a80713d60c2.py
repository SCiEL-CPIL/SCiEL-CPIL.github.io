"""
4-regular graph with girth >= 10.
Combines the Lazebnik-Ustimenko-Woldar (LUW) construction with a cyclic (Z_n-regular)
search strategy. First tries the LUW construction (known to be valid), then attempts
to find smaller graphs via cyclic representation with binary search on n.
"""
import time
import random

def construct(k, g):
    deadline = time.time() + 115  # Leave 5s margin for the 120s limit
    
    # First: try the known LUW construction (always valid, gives a baseline)
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        # Verify it's valid
        if _verify_graph(edges_luw, n_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
        else:
            best_n = float('inf')
            best_edges = None
    else:
        best_n = float('inf')
        best_edges = None
    
    # Now try to find a smaller graph using cyclic construction
    # For a k-regular graph, we need k/2 connection offsets (for k even)
    # The graph is on Z_n, vertex i connects to i+offset_j mod n for j in 0..k/2-1
    # (and symmetrically, i-offset_j mod n)
    
    if k % 2 != 0:
        # Odd degree: more complex, skip cyclic for now
        if best_edges is not None:
            return best_n, best_edges
        # Fallback to LUW
        return _luw_construct(k, g)
    
    half_k = k // 2
    
    # Moore bound for k-regular graph of girth g:
    # For even girth g = 2m: n >= k + k*(k-1) + ... + k*(k-1)^(m-1) * 2
    # For odd girth g = 2m+1: n >= 1 + k + k*(k-1) + ... + k*(k-1)^(m-1)
    moore_bound = _moore_bound(k, g)
    
    # Search for smaller cyclic graphs
    # Start from moore_bound, try increasing n
    # But we know the answer is at most best_n (from LUW)
    
    # Strategy: try a range of n values and for each, search for valid offsets
    # For efficiency, try n values that are multiples of small primes or have nice structure
    
    # Actually, let's be more systematic:
    # For a cyclic k-regular graph on n vertices with offsets o_1, ..., o_{k/2},
    # the graph has girth >= g iff there are no short non-backtracking walks.
    # 
    # A key insight: if n is large relative to the "diameter" of the offset set,
    # the graph will have large girth. But we want small n.
    #
    # Let's try n values from moore_bound to min(best_n, some_upper_limit)
    # and for each, try random offset sets and check girth.
    
    upper_limit = min(best_n - 1, 500)  # Don't search too far
    if upper_limit < moore_bound:
        upper_limit = moore_bound
    
    # For efficiency, focus on n values that are composite (more likely to have structure)
    # and try random offsets
    
    random.seed(42)
    
    for n in range(moore_bound, upper_limit + 1):
        if time.time() > deadline:
            break
        
        # For this n, try several random offset sets
        # Each offset set is a tuple of k/2 distinct non-zero residues mod n
        # (we also implicitly include the negatives)
        
        # The offsets should not include 0, and no offset should be the negative of another
        # (to avoid multi-edges)
        
        # Try up to 50 random offset sets per n
        max_attempts_per_n = 50
        for attempt in range(max_attempts_per_n):
            if time.time() > deadline:
                break
            
            # Generate random offsets
            # Choose k/2 random distinct values from 1..n-1
            # Ensure no pair sums to 0 mod n (to avoid i+o == i-o' mod n for o != o')
            
            candidates = list(range(1, n))
            random.shuffle(candidates)
            
            offsets = []
            used = set()
            for c in candidates:
                if len(offsets) == half_k:
                    break
                if c not in used and (n - c) not in used:
                    offsets.append(c)
                    used.add(c)
                    used.add(n - c)
            
            if len(offsets) < half_k:
                continue
            
            # Build the graph
            edges = []
            edge_set = set()
            for i in range(n):
                for o in offsets:
                    j = (i + o) % n
                    if i < j:
                        key = (i, j)
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append(key)
            
            # Check if it's k-regular and simple
            if len(edges) != n * half_k:
                continue
            
            # Verify girth
            if _verify_graph(edges, n, k, g):
                return n, edges
        
        # If we've spent too much time on this n, move on
        if time.time() > deadline - 10:
            break
    
    # Return the best found
    if best_edges is not None:
        return best_n, best_edges
    
    # Ultimate fallback
    return _luw_construct(k, g)


def _moore_bound(k, g):
    """Compute the Moore bound for a k-regular graph of girth g."""
    if g % 2 == 0:
        # Even girth g = 2m
        m = g // 2
        n = k
        for i in range(1, m):
            n += k * (k - 1) ** i
        n += k * (k - 1) ** (m - 1)  # Last term
        # Actually, for even girth 2m: n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(m-1)) * k / 2
        # Let me recompute: 
        # n >= k + 2 * sum_{i=1}^{m-1} k*(k-1)^i
        # No, the standard formula is:
        # For girth g = 2m: n >= 2 * sum_{i=0}^{m-1} (k-1)^i * k
        # Wait, let me use the standard:
        # n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(m-1)) for even girth 2m? 
        # No. The correct formula:
        # For girth g:
        # If g = 2m (even): n >= 2 * sum_{i=0}^{m-1} (k-1)^i * k
        # Hmm, let me just use: n >= k + k*(k-1) + k*(k-1)^2 + ... 
        # Actually the simplest: BFS from a vertex, no backtracking
        # Level 0: 1 vertex
        # Level 1: k vertices  
        # Level i: k*(k-1)^i vertices for i >= 1
        # For even girth 2m, we need levels 0 to m to be tree-like, but level m has no further expansion
        # So n >= 1 + k + k*(k-1) + k*(k-1)^2 + ... + k*(k-1)^(m-1)
        # Wait, that's for odd girth. For even girth 2m:
        # n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(m-1))
        # No, let me just compute it properly.
        
        # For girth g, the Moore bound is:
        # n >= 1 + k * sum_{i=0}^{floor((g-1)/2) - 1} (k-1)^i  if g is odd
        # n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i  if g is even... no
        
        # Let me use the standard result:
        # For even girth g = 2m: M(k, 2m) = 2 * sum_{i=0}^{m-1} (k-1)^i
        # For odd girth g = 2m+1: M(k, 2m+1) = 1 + k * sum_{i=0}^{m-1} (k-1)^i
        
        m = g // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 2 * s
    else:
        # Odd girth g = 2m + 1
        m = (g - 1) // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 1 + k * s


def _verify_graph(edges, n, k, g):
    """Verify that the graph is k-regular, simple, and has girth >= g."""
    if n <= 0 or len(edges) == 0:
        return False
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False  # Loop
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    # Check k-regular
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False  # Multi-edge
    
    # Check girth using BFS from each vertex
    # For each vertex, do BFS and check for cycles shorter than g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        max_depth = (g - 2) // 2  # We need to check up to this depth
        
        for d in range(1, max_depth + 1):
            new_queue = []
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d
                        parent[v] = u
                        new_queue.append(v)
                    else:
                        # Check for cycle
                        if dist[v] < d - 1:
                            cycle_len = d + dist[v] + 1
                            if cycle_len < g:
                                return False
                        elif dist[v] == d - 1 and parent[u] != v:
                            # This can happen in non-bipartite graphs
                            # Cycle length is 2*d
                            cycle_len = 2 * d
                            if cycle_len < g:
                                return False
            queue = new_queue
    
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
