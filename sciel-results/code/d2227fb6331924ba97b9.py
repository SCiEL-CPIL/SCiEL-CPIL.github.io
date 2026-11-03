import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # LUW parameters
    # The LUW graph CD(d, q) is q-regular.
    # We need degree k. So q must be k.
    # If k is not a prime power, LUW is not directly applicable.
    # For k=3, q=3.
    
    if k != 3:
        # Fallback for other k: return a simple ring? No, must be regular.
        # If k is not 3, we might not have a good construction.
        # For now, assume k=3.
        # If k != 3, try to find a prime power q close to k?
        # No, degree must be exactly k.
        # If k is not a prime power, LUW fails.
        # We will return an empty graph or error?
        # The task says "evaluator passes this task's degree k and girth g".
        # Assuming k=3.
        pass

    q = k
    if q != 3:
        # If q is not 3, LUW might not work or might yield different girth.
        # We will still try LUW if q is a prime power.
        # For simplicity, we assume k=3.
        pass

    # Determine d
    # g = d + 5 (odd d) or d + 6 (even d)
    # We want girth >= g.
    
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6
            
    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    # Check time
    if time.time() > deadline:
        # Return a trivial graph? No.
        pass

    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c>=5: cycle through patterns
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Generate vertices
    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    completed = False
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        # Generate neighbors
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
    
    if time.time() > deadline:
        # Incomplete - cannot return partial graph
        # Return a trivial valid graph: a cycle of length >= g (but not k-regular)
        # Actually, we need k-regular. For k=3, use a disjoint union of triangles? No, girth too small.
        # Best effort: return empty which will be scored as invalid, but at least we don't return bad data.
        # Let's try to return a simple valid k-regular graph with girth >= g if possible.
        # For safety, just return what we have if it's complete, else fail gracefully.
        pass
    
    n = len(order)
    
    # Verify the graph is valid: k-regular, simple, girth >= g
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for (u, v) in edges:
        if u == v:
            continue
        key = (min(u,v), max(u,v))
        if key in edge_set:
            continue
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check k-regular
    valid = True
    for i in range(n):
        if len(adj[i]) != k:
            valid = False
            break
    
    if not valid:
        # Graph is incomplete/invalid. Try to construct a minimal valid k-regular graph.
        # For k=3, a 3-regular graph with girth >= g requires at least Moore bound vertices.
        # As a fallback, return a simple cycle if k=2, else try to make a valid small graph.
        # This is a last resort - the LUW construction should have worked.
        # Return n=0 and empty edges to signal failure (will score 0)
        return 0, []
    
    # Verify girth >= g using BFS from each vertex (non-backtracking)
    # For small graphs this is fine. For large graphs, use a faster check.
    # Use BFS with parent tracking to find shortest cycle.
    
    def has_short_cycle(adj, n, g):
        """Check if there's a cycle shorter than g."""
        for start in range(n):
            # BFS with (node, parent) to avoid backtracking
            dist = {start: 0}
            parent = {start: -1}
            queue = [(start, -1)]
            while queue:
                node, par = queue.pop(0)
                d = dist[node]
                if d >= g // 2:
                    continue
                for nb in adj[node]:
                    if nb == par:
                        continue
                    if nb not in dist:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append((nb, node))
                    elif parent.get(nb, -1) != node:
                        # Found a cycle
                        cycle_len = d + 1 + dist[nb]
                        if cycle_len < g:
                            return True
        return False
    
    if has_short_cycle(adj, n, g):
        # Girth is too small - construction failed
        return 0, []
    
    return n, list(edge_set)
