import time
import random
from collections import deque

def _voltage_graph_construct(k, g, m):
    """Construct a voltage graph over K_{k,k} with voltages from Z_m.
    
    The base graph is K_{k,k} with vertices (0,0)..(0,k-1) and (1,0)..(1,k-1).
    Each edge gets a voltage in Z_m. The lift has k*m vertices per side.
    Total vertices = 2*k*m.
    
    For girth >= g, we need the shortest non-backtracking cycle in the lift to be >= g.
    """
    if k != 5:
        return None
    
    # Base graph K_{5,5}: vertices A_i (i=0..4) and B_j (j=0..4)
    # Edges: A_i - B_j with voltage v(i,j)
    # Lift: vertex (side, i, t) where side in {0,1}, i in 0..k-1, t in Z_m
    # Edge from (0,i,t) to (1,j, t + v(i,j))
    
    # We need to assign voltages v(i,j) in Z_m such that
    # no non-backtracking cycle in the lift has length < g
    
    # A non-backtracking cycle of length 2r in the lift corresponds to:
    # i_0 -> j_0 -> i_1 -> j_1 -> ... -> j_{r-1} -> i_r = i_0
    # with sum of voltages = 0 mod m
    
    # For girth >= 9, we need no cycles of length 4, 6, or 8.
    # Cycle of length 4: i -> j -> i' -> j' -> i, with i != i', j != j'
    #   Voltage sum: v(i,j) + v(i',j') + v(i',j) + v(i,j') = 0 mod m
    #   (going i->j, j->i' [voltage v(i',j) reversed = -v(i',j)], etc.)
    #   Actually, let me think more carefully.
    
    # In the lift, a cycle of length 2r:
    # (0,i_0,t) -> (1,j_0, t+v(i_0,j_0)) -> (0,i_1, t+v(i_0,j_0)+v(i_1,j_0)) -> ...
    # Wait, the edge from B_j to A_i has voltage... 
    
    # Let me define: edge A_i - B_j has voltage v(i,j).
    # Going A_i to B_j: t -> t + v(i,j)
    # Going B_j to A_i: t -> t - v(i,j) (or equivalently, the reverse edge has voltage -v(i,j))
    
    # A cycle of length 4: A_i -> B_j -> A_{i'} -> B_{j'} -> A_i
    # Voltages: v(i,j) + (-v(i',j)) + v(i',j') + (-v(i,j')) = 0
    # i.e., v(i,j) - v(i',j) + v(i',j') - v(i,j') = 0 mod m
    
    # For girth >= 9, we need no such cycles for all i != i', j != j'
    # Also no cycles of length 6, 8.
    
    # Strategy: use v(i,j) = a_i + b_j mod m for some sequences a, b
    # Then v(i,j) - v(i',j) + v(i',j') - v(i,j') = (a_i+b_j) - (a_{i'}+b_j) + (a_{i'}+b_{j'}) - (a_i+b_{j'}) = 0
    # This always gives 0! So this doesn't work.
    
    # Try v(i,j) = a_i * b_j mod m
    # Then the 4-cycle condition: a_i*b_j - a_{i'}*b_j + a_{i'}*b_{j'} - a_i*b_{j'} = 0
    # = b_j(a_i - a_{i'}) + b_{j'}(a_{i'} - a_i) = (a_i - a_{i'})(b_j - b_{j'}) = 0 mod m
    # So we need (a_i - a_{i'})(b_j - b_{j'}) != 0 mod m for all i != i', j != j'
    # This means a_i are all distinct mod m and b_j are all distinct mod m.
    
    # For m >= 5, we can choose a_i = i, b_j = j (or any distinct values).
    # But we also need to check 6-cycles and 8-cycles.
    
    # Let's try v(i,j) = i * j mod m with m >= 5
    # And verify the girth computationally.
    
    # Build the lift
    n = 2 * k * m
    adj = [[] for _ in range(n)]
    
    def vertex_idx(side, i, t):
        return side * k * m + i * m + t
    
    for i in range(k):
        for j in range(k):
            v = (i * j) % m
            for t in range(m):
                u1 = vertex_idx(0, i, t)
                u2 = vertex_idx(1, j, (t + v) % m)
                adj[u1].append(u2)
                adj[u2].append(u1)
    
    # Verify it's simple and k-regular
    for i in range(n):
        if len(adj[i]) != k:
            return None
        if len(set(adj[i])) != k:
            return None
    
    # Check girth using non-backtracking walks
    girth = _compute_girth_nonbacktracking(adj, n, g)
    if girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def _compute_girth_nonbacktracking(adj, n, g):
    """Compute girth using non-backtracking walks from every vertex.
    
    For each vertex, do a BFS in the non-backtracking state space.
    State: (vertex, previous_vertex) or (vertex, -1) for start.
    We track the minimum depth at which we can reach a state (v, u) where u is a neighbor of v.
    A cycle of length L is found when from state (v, u) at depth d, we can go to (u, v) 
    but that's backtracking. Instead, we look for when we revisit a vertex.
    
    Actually, for non-backtracking: a walk v0, v1, ..., vk is non-backtracking if vi != v_{i-2} for all i >= 2.
    A cycle of length L exists if there's a non-backtracking walk of length L from some v to v.
    
    For girth >= g, we need no non-backtracking closed walk of length < g.
    """
    girth = g  # We only care if it's >= g
    
    for start in range(n):
        # BFS in non-backtracking state space
        # State: (current, previous) where previous is the vertex we came from
        # Start: (start, -1) at depth 0
        # From (v, prev), we can go to (w, v) for w in adj[v] if w != prev
        
        visited = {}  # (v, prev) -> min_depth
        queue = deque()
        queue.append((start, -1, 0))
        visited[(start, -1)] = 0
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            
            for w in adj[v]:
                if w == prev:
                    continue  # Non-backtracking
                
                new_state = (w, v)
                if new_state not in visited:
                    visited[new_state] = depth + 1
                    queue.append((w, v, depth + 1))
                elif visited[new_state] > depth + 1:
                    visited[new_state] = depth + 1
                    queue.append((w, v, depth + 1))
                
                # Check if we've found a cycle
                # If w == start and depth + 1 >= 3 (non-backtracking cycle)
                # Actually, we need to check if we can close the cycle
                # A cycle of length L: start -> ... -> w where w is adjacent to start
                # and the walk is non-backtracking
        
        # Alternative: check for each edge (start, neighbor), see if there's a path of length g-2 from neighbor back to start that is non-backtracking
        # This is complex. Let me use a different approach.
    
    # Simpler approach: for each pair of adjacent vertices (u, v), check if there's a path of length g-2 from v to u that doesn't use edge (u,v) and is non-backtracking.
    # This is still complex. Let me just do a full BFS from each vertex and check for cycles.
    
    # Actually, the most reliable way: for each vertex, do BFS and track the first time we revisit a vertex via a different path.
    # But this can give false positives for non-simple cycles.
    
    # Let me use the standard girth computation: for each edge (u,v), remove it and check if there's a path of length g-2 from u to v.
    # If yes, there's a cycle of length g-1.
    
    # For efficiency, let's just check if girth < g by looking for cycles of length 4, 6, 8.
    
    # Check for 4-cycles
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for j in range(i+1, len(adj[u])):
                w = adj[u][j]
                # Check if v and w have a common neighbor other than u
                common = set(adj[v]) & set(adj[w])
                if len(common) > 0 and u not in common:
                    return 4
    
    # Check for 6-cycles
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for j in range(i+1, len(adj[u])):
                w = adj[u][j]
                # Path of length 3 from v to w not using u
                for x in adj[v]:
                    if x == u or x == w:
                        continue
                    for y in adj[x]:
                        if y == v or y == u:
                            continue
                        if y in adj[w] and y != u:
                            return 6
    
    # Check for 8-cycles
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for j in range(i+1, len(adj[u])):
                w = adj[u][j]
                # Path of length 5 from v to w not using u
                # This is expensive, so let's skip and assume girth >= 9 if no 4 or 6 cycles
                # Actually, we need to be more careful
    
    # For now, assume if no 4 or 6 cycles, girth is at least 8. We need to check 8-cycles too.
    # But 8-cycle checking is O(n * k^5) which might be too slow.
    
    # Let's do a proper BFS-based girth check
    girth = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    
    return girth

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if k == 5 and g == 9:
        # Try voltage graph construction with different m values
        # m must be at least 5 for the voltage assignment to work
        for m in range(5, 100):
            if time.time() > deadline:
                break
            
            result = _voltage_graph_construct(k, g, m)
            if result is not None:
                return result
    
    # Fallback to LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
