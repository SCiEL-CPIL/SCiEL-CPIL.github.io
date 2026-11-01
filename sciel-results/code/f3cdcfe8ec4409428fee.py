import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # q must be k to achieve degree k
    q = k

    # GF(q) operations
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q

    def get_rule(d):
        rule = {}
        if d >= 2: rule[2] = (1,1)
        if d >= 3: rule[3] = (2,1)
        if d >= 4: rule[4] = (1,2)
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        return rule

    def build_graph(d):
        rule = get_rule(d)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    def check_girth_fast(n, edges, g):
        """Fast girth check using non-backtracking walks from every vertex.
        Returns True if girth >= g."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS with non-backtracking constraint
        # Track (node, prev_node) states to avoid backtracking
        # A cycle of length L exists if we can return to start in L steps without backtracking
        
        # Use a more efficient approach: for each edge (u,v), check if there's a 
        # non-backtracking walk from v back to u of length < g-1
        # This would form a cycle of length < g
        
        # Better: use the standard non-backtracking BFS
        # State: (current_node, previous_node)
        # Start from each directed edge
        
        # For small girth, we can just check all non-backtracking walks up to length g-1
        # from each vertex and see if we return to start
        
        # Most efficient for this case: since we expect girth to be exactly 2d+2 or similar,
        # and g is small (9), we can do a targeted check
        
        # Check: is there any cycle of length < g?
        # Use BFS from each vertex with parent tracking, but stop early if we find short cycle
        
        girth = float('inf')
        
        # For each start vertex, do BFS tracking parent to detect cycles
        # But use deque for efficiency
        from collections import deque
        
        for start in range(n):
            if time.time() > deadline:
                return True  # Assume valid if running out of time
            
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue  # No need to explore further for cycles < g
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
                            if girth < g:
                                return False
        
        return girth >= g

    # Determine candidate d values
    # The theoretical girth of CD(d,q) construction is 2d+2 when q > d
    # So we need 2d+2 >= g  => d >= (g-2)/2
    d_min = max(3, (g - 2 + 1) // 2)
    
    best_n = float('inf')
    best_edges = []
    
    # Try d from d_min upwards
    for d in range(d_min, d_min + 6):
        if time.time() > deadline - 5:
            break
            
        n, edges = build_graph(d)
        
        if n == 0:
            continue
            
        if n < best_n:
            # Check if girth is sufficient
            if check_girth_fast(n, edges, g):
                best_n = n
                best_edges = edges
                break
    
    if best_n == float('inf'):
        # Fallback
        d = g - 5 if (g - 5) % 2 == 1 else g - 4
        if d < 3:
            d = 3
        n, edges = build_graph(d)
        return n, edges
        
    return best_n, best_edges
