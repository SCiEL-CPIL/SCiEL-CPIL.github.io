import time
import sys
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    # --- Helper: Check girth using BFS from every vertex (optimized) ---
    def check_girth_fast(adj, n, g):
        half_g = g // 2
        max_depth_allowed = (g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] <= max_depth_allowed and dist[v] <= max_depth_allowed:
                            if parent[u] == v or parent[v] == u:
                                continue
                            return False
        return True

    # --- Fallback: LUW construction (Baseline) ---
    def construct_luw(k, g):
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

    # Get baseline
    baseline_n, baseline_edges = construct_luw(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    # --- New Strategy: Deterministic Voltage Lifting on 3-Prism ---
    # The 3-prism graph is 3-regular with 6 vertices.
    # Vertices: 0,1,2 (top triangle), 3,4,5 (bottom triangle).
    # Edges:
    # Top cycle: (0,1), (1,2), (2,0)
    # Bottom cycle: (3,4), (4,5), (5,3)
    # Verticals: (0,3), (1,4), (2,5)
    
    # We lift this graph with Z_m.
    # To ensure girth, we assign voltages such that no short closed walk sums to 0.
    # A linear assignment often works well for large m.
    
    if k == 3:
        base_edges = [
            (0, 1), (1, 2), (2, 0),  # Top
            (3, 4), (4, 5), (5, 3),  # Bottom
            (0, 3), (1, 4), (2, 5)   # Verticals
        ]
        n_base_edges = len(base_edges)
        
        # We want n < baseline_n.
        # n = 6 * m.
        # baseline_n is likely ~936. So m < 156.
        # We try small m first to minimize n.
        
        max_m = baseline_n // 6
        if max_m < 10:
            max_m = 10
            
        search_deadline = deadline - 5.0
        
        # Try deterministic voltage patterns first, then random.
        # Pattern: voltages are linear functions of edge index or position.
        # For a 3-regular graph, we can try voltages like:
        # e0: a, e1: b, e2: c, e3: a, e4: b, e5: c, e6: d, e7: e, e8: f
        # We want to avoid small sums.
        
        for m in range(10, max_m + 1):
            if time.time() > search_deadline:
                break
                
            # Try a few deterministic patterns
            # Pattern 1: All voltages = 1 (will fail for small m, creates 3-cycles if m=3, etc.)
            # Pattern 2: Alternating 1, 2, 3...
            # Pattern 3: Random but seeded for reproducibility within this attempt? 
            # No, let's use a heuristic: large coprime numbers.
            
            # We'll try 5 random assignments for each m.
            # Since m is increasing, the first success is the best n for this strategy.
            # But we might miss a valid smaller m.
            # To mitigate, we try more random attempts for smaller m.
            
            attempts = 10 if m < 50 else 3
            for _ in range(attempts):
                if time.time() > search_deadline:
                    break
                    
                # Random voltages in [1, m-1]
                # Avoid 0 to prevent trivial 1-cycles (though base has none) or small sums.
                voltages = [random.randint(1, m-1) for _ in range(n_base_edges)]
                
                # Build lift
                n = 6 * m
                adj = [[] for _ in range(n)]
                
                for idx, (u, v) in enumerate(base_edges):
                    vol = voltages[idx] % m
                    for i in range(m):
                        u_lift = u * m + i
                        v_lift = v * m + (i + vol) % m
                        adj[u_lift].append(v_lift)
                        adj[v_lift].append(u_lift)
                
                # Verify
                if check_girth_fast(adj, n, g):
                    if n < best_n:
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        best_n = n
                        best_edges = edges
                        # Since m is increasing, we can break if we find a valid graph?
                        # Yes, because smaller m means smaller n.
                        # We are iterating m upwards. The first valid m found is the smallest m found.
                        # We should break the m loop.
                        break
                        
            if best_n < baseline_n:
                # We found an improvement over the baseline.
                # Since we are iterating m upwards, any further m will be larger.
                # We can stop.
                break

    return best_n, best_edges
