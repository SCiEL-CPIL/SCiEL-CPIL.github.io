import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s margin before 150s kill
    
    q = k
    
    def build_luw(d):
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
        
        i = 0
        while i < len(order):
            if time.time() > deadline:
                return None
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth(n, edges, target_girth):
        """
        Check girth >= target_girth using non-backtracking walks from every vertex.
        A cycle of length L exists iff there's a non-backtracking walk of length L 
        that returns to its start. We check for any non-backtracking closed walk 
        of length < target_girth.
        
        For efficiency, we do BFS on the non-backtracking graph: states are (current, prev).
        A non-backtracking walk from (u, v) goes to (v, w) where w != u.
        A cycle of length L >= 3 exists if starting from some (u, v), after L-1 steps 
        we can reach (v, u) ... actually let's think differently.
        
        Simpler: for each directed edge (u,v), do BFS in the non-backtracking graph 
        starting from state (u,v) at depth 1. If we can reach state (v,u) at depth L 
        where L < target_girth, then there's a cycle of length L.
        
        Actually even simpler: a cycle of length L corresponds to a non-backtracking 
        walk of length L that starts and ends at the same vertex. We can check:
        for each vertex s, do non-backtracking BFS and see if we return to s 
        at depth < target_girth.
        
        Let's use: state = (node, prev_node). Start from all (s, -1) for each s? 
        No, non-backtracking walks of length >= 1.
        
        Most efficient for small target_girth: 
        For each directed edge (u,v), we want to know if there's a non-backtracking 
        walk of length L-1 from (u,v) to (v,u) for any L < target_girth.
        This is equivalent to: in the non-backtracking graph, is there a path of 
        length L-1 from node (u,v) to node (v,u)?
        
        Since target_girth is small (9), we can just do BFS up to depth target_girth-1.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length 3, 4, ..., target_girth-1
        # For each possible cycle length L from 3 to target_girth-1:
        # A cycle of length L exists if there's a non-backtracking closed walk of length L.
        # Equivalently, for some directed edge (u,v), there's a non-backtracking walk 
        # of length L-1 from (u,v) to (v,u).
        
        max_L = target_girth - 1
        if max_L < 3:
            return True
            
        # Build non-backtracking adjacency: from (u,v) we can go to (v,w) for w in adj[v], w != u
        # We'll do BFS from each directed edge up to depth max_L - 1
        
        # To be efficient, we do multi-source BFS for each target depth
        # Actually, let's just iterate over all directed edges and do a limited BFS
        
        # For each directed edge (u, v), BFS in non-backtracking graph
        # State: (current, prev)
        # Start: (u, v) at depth 1
        # We want to reach (v, u) at depth L for L in [3, max_L]
        
        # More efficient: for each starting vertex s, do non-backtracking BFS
        # and track if we return to s at depth < target_girth
        
        for s in range(n):
            # Non-backtracking BFS from vertex s
            # State: (node, prev) where prev is the previous node in the walk
            # Start: (s, -1) means we're at s with no previous (start of walk)
            # From (s, -1), we can go to (neighbor, s) for each neighbor
            # From (node, prev), we can go to (next, node) for next in adj[node], next != prev
            
            # We want to find if there's a non-backtracking walk from s back to s 
            # of length L where 3 <= L < target_girth
            
            # BFS: visited[(node, prev)] = length of shortest non-backtracking walk 
            # ending at (node, prev) starting from s
            visited = {}
            # Start states: we're at s, having come from "nowhere"
            # Represent as (s, -1)
            queue = deque()
            queue.append((s, -1, 0))  # (node, prev, length)
            visited[(s, -1)] = 0
            
            while queue:
                node, prev, length = queue.popleft()
                
                if length >= target_girth - 1:
                    continue
                    
                for next_node in adj[node]:
                    if next_node == prev:
                        continue
                    new_state = (next_node, node)
                    new_length = length + 1
                    
                    if next_node == s and new_length >= 3 and new_length < target_girth:
                        # Found a cycle of length new_length
                        return False
                    
                    if new_state not in visited:
                        visited[new_state] = new_length
                        if new_length < target_girth - 1:
                            queue.append((next_node, node, new_length))
        
        return True

    # Estimate minimal d
    # Girth of LUW(d, q) is 2d+2
    # We need girth >= g
    # If g is odd, we need girth >= g+1 (since girth is even)
    # So 2d+2 >= g+1 => 2d >= g-1 => d >= (g-1)/2
    # If g is even, we need girth >= g
    # So 2d+2 >= g => 2d >= g-2 => d >= (g-2)/2
    
    if g % 2 == 1:
        d_min = (g - 1) // 2
    else:
        d_min = (g - 2) // 2
        
    d_min = max(3, d_min)
    
    best_n = float('inf')
    best_edges = None
    
    # Try d from d_min upwards
    # Limit d to avoid explosion
    max_d = d_min + 2
    
    for d in range(d_min, max_d + 1):
        if time.time() > deadline:
            break
            
        result = build_luw(d)
        if result is None:
            break
            
        n, edges = result
        
        # Check girth
        if check_girth(n, edges, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            # Larger d gives larger n. So minimal d that works is best.
            break
            
    if best_edges is None:
        # Fallback: try slightly larger d
        d = d_min + 1
        result = build_luw(d)
        if result:
            n, edges = result
            if check_girth(n, edges, g):
                return n, edges
        return 0, []
        
    return best_n, best_edges
