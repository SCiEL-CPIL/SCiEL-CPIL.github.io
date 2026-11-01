"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 145  # Leave 5s safety margin
    
    def check_girth_fast(adj, n, target_girth):
        """Check if graph has girth >= target_girth using non-backtracking BFS."""
        max_depth = (target_girth - 1) // 2
        
        for start in range(n):
            # visited[depth][vertex] = parent vertex, -1 if unvisited, -2 for start
            visited = [[-1] * n for _ in range(max_depth + 1)]
            visited[0][start] = -2
            
            queue = deque([(start, 0, -1)])
            
            while queue:
                v, depth, parent = queue.popleft()
                if depth >= max_depth:
                    continue
                
                for u in adj[v]:
                    if u == parent:
                        continue
                    if visited[depth + 1][u] != -1:
                        return False
                    visited[depth + 1][u] = v
                    queue.append((u, depth + 1, v))
        
        return True
    
    def build_lift(base_edges, voltages, m, k_val):
        """Build the voltage graph lift efficiently."""
        n_b = max(v for e in base_edges for v in e) + 1
        n = n_b * m
        adj = [[] for _ in range(n)]
        
        for (u, v), vol in zip(base_edges, voltages):
            vol_mod = vol % m
            for x in range(m):
                u_lift = u * m + x
                v_lift = v * m + ((x + vol_mod) % m)
                adj[u_lift].append(v_lift)
        
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != k_val:
                return None, None
        
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges
    
    # Generate base graphs
    def get_base_graphs(k_val):
        """Generate several k-regular base graphs."""
        graphs = []
        
        # K_{k+1} if k+1 <= 10
        if k_val + 1 <= 10:
            edges = []
            for i in range(k_val + 1):
                for j in range(i + 1, k_val + 1):
                    edges.append((i, j))
            graphs.append(edges)
        
        # Petersen graph for k=3 (but we need k=5, so use other structures)
        # For k=5: Use a circulant-like structure or complete bipartite
        
        # 5-regular graph on 10 vertices: K_{5,5} is 5-regular
        if k_val == 5:
            edges = []
            for i in range(5):
                for j in range(5):
                    edges.append((i, 5 + j))
            graphs.append(edges)
            
            # Another 5-regular graph: complement of 5 disjoint edges on 10 vertices
            # Actually, let's use a different structure: two K6 with cross connections
            # Or simply: 5-regular graph on 6 vertices is K6, already have that
            
            # Try a 5-regular graph on 12 vertices: each vertex connected to 5 others
            # Use a structured approach
            edges = []
            n_b = 12
            for i in range(n_b):
                for offset in [1, 2, 3, 4, 5]:
                    j = (i + offset) % n_b
                    if i < j:
                        edges.append((i, j))
            # Check if this is 5-regular
            deg = [0] * n_b
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == k_val for d in deg):
                graphs.append(edges)
        
        return graphs
    
    base_graphs = get_base_graphs(k)
    
    best_n = float('inf')
    best_edges = None
    
    random.seed(42)
    
    # Try each base graph
    for base_edges in base_graphs:
        if time.time() > deadline:
            break
        
        n_b = max(v for e in base_edges for v in e) + 1
        num_base_edges = len(base_edges)
        
        # For n = n_b * m, we want n to be reasonable
        # Moore bound for (5,9) is around 362, so m should be ~362/n_b
        min_m = max(1, (300 + n_b - 1) // n_b)  # Aim for n >= 300
        max_m = min(200, (1500 + n_b - 1) // n_b)  # Don't go too large
        
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            
            n = n_b * m
            
            # Try several voltage assignments
            num_trials = 30 if m < 100 else 15
            
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                # Random voltages
                voltages = [random.randint(0, m - 1) for _ in range(num_base_edges)]
                
                n_lift, edges = build_lift(base_edges, voltages, m, k)
                if n_lift is None:
                    continue
                
                if n_lift >= best_n:
                    continue
                
                # Build adjacency for girth check
                adj = [[] for _ in range(n_lift)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                if check_girth_fast(adj, n_lift, g):
                    best_n = n_lift
                    best_edges = edges
                    if n_lift < 450:
                        return n_lift, edges
    
    # Fallback to LUW construction
    if best_edges is not None:
        return best_n, best_edges
    
    # Original LUW construction as fallback
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
# EVOLVE-BLOCK-END
