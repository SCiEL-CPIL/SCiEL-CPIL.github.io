import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # For k=5, g=9, try L-U-W with d=4 first
    if k == 5 and g == 9:
        # d=4 gives size 2*5^4 = 1250
        # We want the smallest n, so try d=4 first
        n, edges = _construct_luw(k, 4)
        # Check if girth is sufficient
        if _girth_at_least(n, edges, g):
            return n, edges
        
        # If d=4 fails girth, try d=5
        if time.time() < deadline - 10:
            n, edges = _construct_luw(k, 5)
            if _girth_at_least(n, edges, g):
                return n, edges
        
        # If d=5 fails, try d=6
        if time.time() < deadline - 5:
            n, edges = _construct_luw(k, 6)
            if _girth_at_least(n, edges, g):
                return n, edges
        
        # Fallback: use larger d to ensure girth
        d = 7
        n, edges = _construct_luw(k, d)
        return n, edges

    # General case: use seed logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _construct_luw(q, d)

def _construct_luw(q, d):
    """
    Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q).
    """
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
        r = (c - 5) % 4
        c0 = c - r
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _girth_at_least(n, edges, g):
    """
    Check if the girth of the graph is at least g.
    Returns True if no cycle of length < g exists.
    Uses non-backtracking BFS from each vertex to certify girth.
    """
    if g <= 1:
        return True
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each start vertex, do non-backtracking BFS
    # Track (vertex, previous_vertex) states
    # If we reach a state where vertex == start and depth >= 2, we found a cycle
    # Actually, we need to detect any cycle of length < g
    
    for start in range(n):
        # State: (current_vertex, prev_vertex)
        # dist[(v, p)] = length of non-backtracking walk from (start, None) to (v, p)
        # We want to find if there's a cycle of length < g
        
        # BFS from (start, -1) where -1 means no previous
        # dist[(v, p)] = shortest non-backtracking path length from start to v coming from p
        dist = {}
        queue = [(start, -1, 0)]  # (vertex, prev, depth)
        dist[(start, -1)] = 0
        head = 0
        
        while head < len(queue):
            u, prev_u, depth = queue[head]
            head += 1
            
            if depth >= g - 1:
                continue
            
            for v in adj[u]:
                if v == prev_u:
                    continue  # Non-backtracking
                
                new_depth = depth + 1
                state = (v, u)
                
                if state in dist:
                    # We've seen this state before
                    # Check if this creates a short cycle
                    # The cycle would be: start -> ... -> u -> v -> ... -> (v, u) already visited
                    # Actually, if we revisit (v, u), it means there's a cycle
                    # The cycle length is at least 2 (but we skip backtracking, so min is 3)
                    # We need to be careful here
                    
                    # If v == start and new_depth >= 2, we found a cycle of length new_depth
                    if v == start and new_depth >= 2:
                        if new_depth < g:
                            return False
                else:
                    dist[state] = new_depth
                    queue.append((v, u, new_depth))
                    
                    # If we reach start with depth >= 2, we found a cycle
                    if v == start and new_depth >= 2:
                        if new_depth < g:
                            return False
    
    return True
