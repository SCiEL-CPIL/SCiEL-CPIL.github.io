import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # The strategy is to use a voltage graph on K6.
    # This only works for k=5.
    if k != 5:
        # Fallback to LUW for other k
        return luw_construct(k, g)
        
    if g < 3:
        return 0, []
        
    # Base graph is K6
    base_nodes = list(range(6))
    base_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
    
    # Pre-compute all cycles of length < g in K6
    # We need cycles of length 3 to g-1
    max_cycle_len = g - 1
    if max_cycle_len < 3:
        # Trivial case, girth < 3 is impossible for simple graph with edges
        return 6, [] # Just K6? No, girth must be >= g. If g=3, K6 has girth 3.
        
    cycles = []
    # Find all simple cycles of length 3 to max_cycle_len
    # Since K6 is small, we can do DFS
    def find_cycles(start, current, path, visited):
        if len(path) > 2 and path[0] == current:
            l = len(path) - 1
            if 3 <= l <= max_cycle_len:
                # Normalize cycle to avoid duplicates
                # Rotate to start with min node
                min_idx = path.index(min(path))
                norm_path = tuple(path[min_idx:] + path[:min_idx])
                # Check reverse
                rev_path = tuple(norm_path[0] + norm_path[:0:-1])
                if norm_path < rev_path:
                    cycles.append(norm_path)
            return
            
        for neighbor in range(6):
            if neighbor == current:
                continue
            if neighbor == start and len(path) < 3:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                find_cycles(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
            elif neighbor == start and len(path) >= 3:
                find_cycles(start, neighbor, path, visited)
                
    for s in range(6):
        find_cycles(s, s, [s], {s})
        
    # Remove duplicates
    cycles = list(set(cycles))
    
    # Search for m and voltages
    # Moore bound for k=5, g=9 is 426. n=6m. m >= 71.
    # Let's start m from 72
    min_m = 72
    if g > 9:
        # Adjust min_m for larger g
        # Moore bound grows roughly (k-1)^((g-1)/2)
        # For g=11, bound is larger.
        # Let's just start from 72 and go up.
        pass
        
    best_n = float('inf')
    best_edges = None
    
    for m in range(min_m, 150):
        if time.time() > deadline:
            break
            
        # Try to find voltages for this m
        found = False
        # Number of restarts
        max_restarts = 500
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Assign random voltages
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
                
            # Check constraints
            valid = True
            for cyc in cycles:
                # Compute sum of voltages along cycle
                s = 0
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    if u < v:
                        s += voltages[(u, v)]
                    else:
                        s -= voltages[(v, u)]
                if s % m == 0:
                    valid = False
                    break
                    
            if valid:
                found = True
                break
                
        if found:
            # Construct the graph
            n = 6 * m
            edges = []
            for u, v in base_edges:
                w = voltages[(u, v)]
                for i in range(m):
                    # Edge (u, i) to (v, i+w)
                    u_node = u * m + i
                    v_node = v * m + (i + w) % m
                    edges.append((u_node, v_node))
                    
            # Verify girth (optional but good for safety)
            # If verification fails, continue search
            if verify_girth(n, edges, g):
                return n, edges
            else:
                # Should not happen if logic is correct, but just in case
                continue
                
    # If no solution found with voltage method, fallback to LUW
    return luw_construct(k, g)

def luw_construct(k, g):
    # Original LUW code
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

def verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each node to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
