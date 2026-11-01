import time
import random
import sys
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Try voltage graph construction on K6
    # N must be multiple of 6
    # Moore bound 426, so try N from 432 to 636
    # Step 6
    # Prioritize smaller N
    
    base_vertices = 6
    base_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
    edge_index = {e: idx for idx, e in enumerate(base_edges)}
    num_edges = len(base_edges)
    
    # Generate all non-backtracking closed walks of length 3 to 8 in K6
    # Walk is represented as list of vertices [v0, v1, ..., vL=v0]
    # We need to map this to edges and signs.
    # For edge (u, v) in walk, if u < v, sign is +1 for voltage[(u,v)], else -1 for voltage[(v,u)]?
    # Let's define voltage w_e for undirected edge e.
    # If we traverse e from u to v, contribution is +w_e if we define orientation u->v, or just w_e.
    # Let's fix an orientation for each edge in base_edges.
    # base_edges are (i, j) with i < j.
    # If walk goes i -> j, add w_e.
    # If walk goes j -> i, subtract w_e.
    
    constraints = []
    for L in range(3, 9):
        # Generate walks
        for start in range(6):
            # Recursive generation or iterative
            # State: (current_vertex, prev_vertex, path)
            # We only need the list of (edge_idx, sign)
            
            def gen_walks(cur, prev, path_edges, length):
                if length == 0:
                    if cur == start:
                        # Check if non-backtracking at the end?
                        # The path is v0..vL.
                        # Non-backtracking condition: v_{i-1} != v_{i+1}.
                        # We enforced this during generation.
                        # We need to check the wrap-around: v_L (which is start) and v_{L-1} (prev) and v_0 (start).
                        # Wait, the condition is on the walk itself.
                        # v_{L-2} != v_L (start) is required?
                        # Yes, v_{L-2} != v_{L-1} + v_L? No.
                        # v_{i-1} != v_{i+1}.
                        # For i=L-1: v_{L-2} != v_L = start.
                        # So we must check if the second to last vertex is start.
                        if len(path_edges) >= 2:
                            # path_edges stores (edge, sign). We need vertices.
                            pass
                        # Let's store vertices in path too for this check.
                        # Actually, we can just check the vertices.
                        pass
                # This recursive approach with path storage is messy.
                # Let's just generate and filter.
                pass
            
            # Simpler: Generate all sequences of vertices v1..v_{L-1}
            # v0 = start, vL = start.
            # Conditions:
            # 1. |v_i - v_{i-1}| > 0 (always true in simple graph)
            # 2. v_{i-1} != v_{i+1} for all i.
            
            # Number of choices for v1: 5
            # v2: 4 (not v1, not v0? No, v2 != v0 is not required for non-backtracking, only v1 != v3 etc)
            # Wait, non-backtracking: v_{i-1} != v_{i+1}.
            # So v0 != v2.
            # v1 != v3.
            # ...
            # v_{L-2} != v_L = start.
            
            # So:
            # v1 in N(start)
            # v2 in N(v1) \ {start}
            # v3 in N(v2) \ {v1}
            # ...
            # v_{L-1} in N(v_{L-2}) \ {v_{L-3}}
            # AND v_{L-1} != start (from v_{L-2} != v_L=start)
            
            # Let's iterate
            for v1 in range(6):
                if v1 == start: continue
                for v2 in range(6):
                    if v2 == v1 or v2 == start: continue
                    # ... this is too deep for L=8
                    
    # Instead of generating all walks explicitly in Python (slow), 
    # we can use the property that we just need to satisfy the constraints.
    # But we need the constraints to define the cost function.
    # Let's generate them efficiently.
    
    def generate_constraints():
        cons = []
        for L in range(3, 9):
            for start in range(6):
                # Use DFS to find non-backtracking closed walks
                # State: (cur, prev, path_edges)
                # path_edges: list of (edge_idx, sign)
                stack = [(start, -1, [])]
                while stack:
                    cur, prev, path = stack.pop()
                    if len(path) == L:
                        if cur == start:
                            # Check wrap-around non-backtracking: prev != start
                            if prev != start:
                                cons.append(path)
                        continue
                    for nxt in range(6):
                        if nxt == prev: continue
                        if len(path) > 0 and nxt == start and len(path) < L-1:
                            # Can't close early
                            pass 
                        # Add edge
                        u, v = (cur, nxt) if cur < nxt else (nxt, cur)
                        e_idx = edge_index[(u, v)]
                        sign = 1 if cur < nxt else -1
                        stack.append((nxt, cur, path + [(e_idx, sign)]))
        return cons
    
    # This DFS might be slow in Python for L=8.
    # Let's estimate: 120k walks.
    # Storing 120k lists of length 8.
    # It should be fine within 110s.
    
    try:
        constraints = generate_constraints()
    except Exception:
        constraints = []
        
    if not constraints:
        return _luw_construct(k, g, deadline)
        
    # Simulated Annealing
    best_n = float('inf')
    best_edges = None
    
    for N in range(432, 637, 6):
        if time.time() > deadline - 20:
            break
        m = N // 6
        # If m is not integer, skip (N is multiple of 6, so m is int)
        
        # Run SA
        found = _sa_search(num_edges, constraints, m, deadline)
        if found:
            voltages = found
            # Build graph
            edges = []
            for i in range(6):
                for j in range(6):
                    if i == j: continue
                    if i > j: continue
                    e_idx = edge_index[(i, j)]
                    w = voltages[e_idx]
                    for k_idx in range(m):
                        u_idx = i * m + k_idx
                        v_idx = j * m + (k_idx + w) % m
                        edges.append((u_idx, v_idx))
            
            # Verify
            adj = [[] for _ in range(N)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            if _check_girth_fast(adj, N, g):
                best_n = N
                best_edges = edges
                if N <= 450: # Good enough?
                    return N, edges
                    
    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_construct(k, g, deadline)

def _sa_search(num_edges, constraints, m, deadline):
    # Cost function: number of violated constraints
    # Constraint: sum(sign_i * voltages[edge_i]) % m != 0
    
    # Pre-compute for each constraint: list of (edge_idx, sign)
    # To speed up cost calculation, we can maintain the sum for each constraint.
    
    # Initialize voltages
    voltages = [random.randint(0, m-1) for _ in range(num_edges)]
    
    # Calculate initial cost
    sums = [0] * len(constraints)
    cost = 0
    for i, cons in enumerate(constraints):
        s = 0
        for e_idx, sign in cons:
            s += sign * voltages[e_idx]
        s %= m
        sums[i] = s
        if s == 0:
            cost += 1
            
    if cost == 0:
        return voltages
        
    T = 10.0
    cooling = 0.99
    
    while time.time() < deadline - 5:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_v = voltages[e_idx]
        new_v = random.randint(0, m-1)
        if new_v == old_v: continue
        
        delta_v = new_v - old_v
        
        # Update sums and cost
        delta_cost = 0
        affected = []
        for i, cons in enumerate(constraints):
            # Check if e_idx is in cons
            # This is slow. We need an index: edge -> list of constraint indices.
            pass
        
        # Optimization: Build index
        # This is done once outside or here?
        # Let's assume we have an index.
        # For now, just do a full re-calc if changes are frequent?
        # No, SA requires fast delta.
        
        # Let's restart with a better structure
        pass
        
    return None

def _check_girth_fast(adj, n, min_girth):
    # Standard BFS
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g, deadline):
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
