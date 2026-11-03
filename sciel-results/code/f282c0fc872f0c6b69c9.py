import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Fallback LUW Construction
    def luw_construct(k, g):
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore Bound
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore_bound = k * sum((k - 1) ** i for i in range(r)) + (k - 1) ** r

    # Strategy: Use Voltage Graph Lift of W(3,4) Incidence Graph
    # This is only applicable for k=5, g=9 (or similar small cases)
    if k == 5 and g == 9:
        try:
            n_lift, edges_lift = construct_lift_w34(13) # Try m=13 first
            if n_lift < best_n:
                best_n = n_lift
                best_edges = edges_lift
        except:
            pass

    return best_n, best_edges

def construct_lift_w34(m):
    """
    Construct a 5-regular graph of girth >= 9 using a voltage graph lift
    of the incidence graph of the symplectic polar space W(3,4).
    """
    # 1. Construct W(3,4) Incidence Graph
    # Field GF(4)
    # Elements: 0, 1, 2, 3 where 2 = omega, 3 = omega^2
    # Add table:
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=1, 2+3=0
    # 3+3=2
    add_t = [
        [0,1,2,3],
        [1,0,3,2],
        [2,3,1,0],
        [3,2,0,1]
    ]
    # Mul table:
    # 0*x=0
    # 1*x=x
    # 2*1=2, 2*2=3, 2*3=1
    # 3*1=3, 3*2=1, 3*3=2
    mul_t = [
        [0,0,0,0],
        [0,1,2,3],
        [0,2,3,1],
        [0,3,1,2]
    ]
    
    def f_add(a, b): return add_t[a][b]
    def f_mul(a, b): return mul_t[a][b]
    
    # Symplectic form B(x,y) = x1y2 + x2y1 + x3y4 + x4y3
    def symplectic(x, y):
        return f_add(f_add(f_mul(x[0], y[1]), f_mul(x[1], y[0])), 
                     f_add(f_mul(x[2], y[3]), f_mul(x[3], y[2])))
    
    # Isotropic vectors (non-zero)
    points = []
    for x0 in range(4):
        for x1 in range(4):
            for x2 in range(4):
                for x3 in range(4):
                    if x0==0 and x1==0 and x2==0 and x3==0: continue
                    if symplectic([x0,x1,x2,x3], [x0,x1,x2,x3]) == 0:
                        points.append((x0,x1,x2,x3))
                        
    # Points in W(3,4) are 1-dimensional subspaces.
    # Since GF(4) has 3 non-zero elements, each point has 3 representatives.
    # We pick the canonical representative: first non-zero element is 1.
    unique_points = []
    seen = set()
    for p in points:
        # Normalize
        idx = 0
        while p[idx] == 0: idx += 1
        inv = 1
        if p[idx] == 2: inv = 3
        elif p[idx] == 3: inv = 2
        norm = tuple(f_mul(p[i], inv) for i in range(4))
        if norm not in seen:
            seen.add(norm)
            unique_points.append(norm)
            
    # Lines in W(3,4) are 2-dimensional isotropic subspaces.
    # A line is determined by two independent isotropic points p, q such that B(p,q)=0.
    unique_lines = []
    seen_lines = set()
    for i in range(len(unique_points)):
        for j in range(i+1, len(unique_points)):
            p = unique_points[i]
            q = unique_points[j]
            if symplectic(list(p), list(q)) == 0:
                # The line is spanned by p and q.
                # Canonical representation: sort the two points?
                # Or just use the set of points on the line.
                # A line in W(3,4) has q+1 = 5 points.
                # Let's find all points on the line.
                line_points = set()
                for a in range(4):
                    for b in range(4):
                        # r = a*p + b*q
                        r = tuple(f_add(f_mul(p[i], a), f_mul(q[i], b)) for i in range(4))
                        if r == (0,0,0,0): continue
                        # Normalize r
                        idx = 0
                        while r[idx] == 0: idx += 1
                        inv = 1
                        if r[idx] == 2: inv = 3
                        elif r[idx] == 3: inv = 2
                        rn = tuple(f_mul(r[i], inv) for i in range(4))
                        line_points.add(rn)
                
                # The line is defined by the set of 5 points.
                # Use frozenset as key.
                key = frozenset(line_points)
                if key not in seen_lines:
                    seen_lines.add(key)
                    unique_lines.append(list(line_points))
                    
    # Build Incidence Graph G0
    # Vertices: 0..16 for points, 17..33 for lines
    n_points = len(unique_points)
    n_lines = len(unique_lines)
    n0 = n_points + n_lines
    
    adj0 = [set() for _ in range(n0)]
    point_index = {p: i for i, p in enumerate(unique_points)}
    
    for l_idx, line in enumerate(unique_lines):
        for p in line:
            p_idx = point_index[p]
            adj0[p_idx].add(n_points + l_idx)
            adj0[n_points + l_idx].add(p_idx)
            
    # Verify G0 is 5-regular
    for i in range(n0):
        if len(adj0[i]) != 5:
            raise ValueError("G0 not 5-regular")
            
    # 2. Find all cycles of length < 9 in G0
    # G0 is bipartite, so only even cycles.
    # Cycles of length 4, 6, 8.
    # We need to break all cycles of length 4, 6, 8.
    # Actually, if we break 4 and 6, do we break 8?
    # An 8-cycle might not contain a 4 or 6 cycle.
    # So we must find all 4, 6, 8 cycles.
    
    cycles = []
    for start in range(n0):
        # BFS/DFS to find cycles
        # Since G0 is small, we can use a simple backtracking search for cycles
        # up to length 8.
        pass # Placeholder: Implement cycle finding
        
    # For efficiency, we can just check the voltage sum for all cycles found.
    # But we need the list of cycles.
    # Let's use a standard cycle enumeration algorithm for small graphs.
    
    def find_cycles(adj, max_len):
        cycles = []
        n = len(adj)
        for start in range(n):
            # DFS
            stack = [(start, [start])]
            while stack:
                u, path = stack.pop()
                if len(path) > max_len: continue
                if len(path) >= 3:
                    if path[-1] == start:
                        # Found a cycle
                        # Normalize: start with smallest vertex
                        min_idx = min(range(len(path)), key=lambda i: path[i])
                        # Rotate
                        cyc = path[min_idx:] + path[:min_idx]
                        # Remove last (start)
                        cyc = cyc[:-1]
                        # Ensure no duplicates (reverse)
                        # Check if already in list
                        if cyc not in cycles and list(reversed(cyc)) not in cycles:
                            cycles.append(cyc)
                else:
                    for v in adj[u]:
                        if v not in path:
                            stack.append((v, path + [v]))
        return cycles
        
    # This DFS is slow. Better to use BFS for each start vertex to find cycles.
    # Or just hardcode the structure? No.
    # Let's use a more efficient cycle finder for small graphs.
    
    # Actually, for the lift to have girth >= 9, we need:
    # For every cycle C in G0 of length L < 9, the sum of voltages != 0.
    # We can assign voltages randomly and check.
    
    edges0 = []
    for u in range(n0):
        for v in adj0[u]:
            if u < v:
                edges0.append((u, v))
                
    # Map edge to index
    edge_idx = {(u,v): i for i, (u,v) in enumerate(edges0)}
    
    # Find all cycles of length 4, 6, 8
    # We'll use a simple backtracking with pruning.
    # Given n0=34, this is small.
    
    def get_cycles(adj, max_len):
        n = len(adj)
        cycles = []
        for start in range(n):
            # BFS from start
            # dist, parent
            # We want cycles that start and end at start.
            # This is equivalent to finding paths from start to start of length L.
            # We can use DFS.
            def dfs(u, path, visited):
                if len(path) > max_len: return
                if len(path) >= 3:
                    if u == start:
                        # Cycle found
                        cyc = path[:]
                        # Normalize
                        min_val = min(cyc)
                        min_idx = cyc.index(min_val)
                        cyc = cyc[min_idx:] + cyc[:min_idx]
                        cyc = cyc[:-1] # Remove start
                        # Check duplicates
                        if cyc not in cycles:
                            # Check reverse
                            rev = list(reversed(cyc))
                            if rev not in cycles:
                                cycles.append(cyc)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs(v, path, visited)
                        path.pop()
                        visited.remove(v)
            
            dfs(start, [start], {start})
        return cycles
        
    all_cycles = get_cycles(adj0, 8)
    
    # 3. Solve for voltages
    # We need to assign v_e in Z_m such that sum_{e in C} v_e != 0 for all C in all_cycles.
    
    def solve_voltages(m, max_iter=1000):
        n_edges = len(edges0)
        # Random initialization
        voltages = [random.randint(0, m-1) for _ in range(n_edges)]
        
        for _ in range(max_iter):
            # Check all cycles
            valid = True
            for C in all_cycles:
                # C is a list of vertices
                # Edges are (C[i], C[i+1])
                s = 0
                for i in range(len(C)):
                    u, v = C[i], C[(i+1)%len(C)]
                    if u > v: u, v = v, u
                    s = (s + voltages[edge_idx[(u,v)]]) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                return voltages
            
            # Perturb
            # Pick a random cycle that failed? Or just random edge.
            idx = random.randint(0, n_edges-1)
            voltages[idx] = random.randint(0, m-1)
            
        return None
        
    voltages = solve_voltages(m)
    if voltages is None:
        return None, None
        
    # 4. Construct Lift Graph
    # Vertices: (u, i) where u in 0..n0-1, i in 0..m-1
    n_lift = n0 * m
    edges_lift = []
    
    for u in range(n0):
        for v in adj0[u]:
            if u < v:
                e_idx = edge_idx[(u,v)]
                vol = voltages[e_idx]
                # Edge from (u, i) to (v, i+vol)
                # And from (v, i) to (u, i+vol) ?
                # Voltage graph lift:
                # Edge (u,v) with voltage vol becomes edges:
                # (u, i) -- (v, i+vol) for all i.
                # Since graph is undirected, we add (u, i) to (v, i+vol).
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + (i + vol) % m
                    edges_lift.append((u_lift, v_lift))
                    
    # Verify
    # Check regularity
    adj_lift = [set() for _ in range(n_lift)]
    for u, v in edges_lift:
        adj_lift[u].add(v)
        adj_lift[v].add(u)
        
    for i in range(n_lift):
        if len(adj_lift[i]) != 5:
            return None, None
            
    # Check girth
    # BFS from each vertex up to depth 4 (since girth >= 9, no cycles < 9)
    # Max depth 4 means we check paths of length 5.
    # If we find a cycle of length < 9, it must be detected.
    # Actually, just check if any cycle < 9 exists.
    
    def check_girth(adj, n, g):
        max_depth = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    elif v != s and dist[v] <= du:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
        
    if not check_girth(adj_lift, n_lift, 9):
        return None, None
        
    return n_lift, edges_lift

# Note: The above code is a sketch. The cycle finding and voltage solving need to be robust.
# Given the time limit, I will provide a complete, executable version.
