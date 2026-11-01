"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque
    import time

    def check_girth_fast(m, voltages, required_girth):
        """Check if the lifted K_{3,3} graph has girth >= required_girth using BFS with early termination."""
        n = 6 * m
        # Vertex encoding: side * 3 * m + base * m + shift
        # Precompute neighbors for each vertex type
        # For vertex (side, base, shift), neighbors are:
        # If side=0 (A-side): neighbors are (1, j, (shift + voltages[base][j]) % m) for j in 0..2
        # If side=1 (B-side): neighbors are (0, i, (shift - voltages[i][base]) % m) for i in 0..2
        
        # Build adjacency on the fly using the voltage structure
        # Instead of storing full adjacency, compute neighbors as needed
        
        half_girth = required_girth // 2
        
        for start in range(n):
            # Decode start
            start_side = start // (3 * m)
            start_base = (start % (3 * m)) // m
            start_shift = start % m
            
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_girth:
                    continue
                u_side = u // (3 * m)
                u_base = (u % (3 * m)) // m
                u_shift = u % m
                
                for v in range(3):
                    if u_side == 0:
                        # From A-side vertex (0, u_base, u_shift) to B-side
                        v_shift = (u_shift + voltages[u_base][v]) % m
                        neighbor = 1 * 3 * m + v * m + v_shift
                    else:
                        # From B-side vertex (1, u_base, u_shift) to A-side
                        v_shift = (u_shift - voltages[v][u_base]) % m
                        neighbor = 0 * 3 * m + v * m + v_shift
                    
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[u] + 1
                        parent[neighbor] = u
                        queue.append(neighbor)
                    elif neighbor != parent[u]:
                        cycle_len = dist[u] + dist[neighbor] + 1
                        if cycle_len < required_girth:
                            return False
        
        return True

    def build_edges(m, voltages):
        """Build the edge list for the lifted graph."""
        n = 6 * m
        edges = []
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = 0 * 3 * m + i * m + s
                    w = 1 * 3 * m + j * m + (s + v) % m
                    edges.append((u, w))
        return n, edges

    # Compute Moore bound
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    # Time budget
    start_time = time.time()
    time_budget = 110  # seconds
    
    random.seed(42)
    
    # Try m values from m_min upward
    for m in range(m_min, min(m_min + 300, 500)):
        if 6 * m > 3000:  # Cap to avoid too large graphs
            break
        
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Try multiple random voltage assignments
        num_trials = 30
        for trial in range(num_trials):
            if time.time() - start_time > time_budget:
                break
            
            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth_fast(m, voltages, g):
                n, edges = build_edges(m, voltages)
                return n, edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
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
