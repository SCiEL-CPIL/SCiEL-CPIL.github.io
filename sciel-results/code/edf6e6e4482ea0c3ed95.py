# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # LUW Construction (Lazebnik-Ustimenko-Woldar)
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
            if g <= 5:
                d = 3
            else:
                d = 3
                while True:
                    girth_est = d + 5 if d % 2 == 1 else d + 4
                    if girth_est >= g:
                        break
                    d += 1

        # Define field operations
        if q == 4:
            # GF(4) = {0, 1, 2, 3}
            # Addition is XOR
            # Multiplication table:
            # 0*0=0, 0*1=0, 0*2=0, 0*3=0
            # 1*0=0, 1*1=1, 1*2=2, 1*3=3
            # 2*0=0, 2*1=2, 2*2=3, 2*3=1
            # 3*0=0, 3*1=3, 3*2=1, 3*3=2
            mul_table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            # Prime field
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q

        # Define the rule for LUW graph
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        for c in range(2, d + 1):
            rule_a[c], rule_b[c] = rule[c]

        # Precompute neighbor lists for all possible vertices
        # Vertex is encoded as (side, x0, x1, ..., xd)
        # We use a dictionary to map vertex tuples to indices
        # Since q and d are small, the total number of vertices is 2 * q^(d+1)
        
        # For q=4, d=5, total vertices = 2 * 4^6 = 8192. This is small.
        # We can precompute all neighbors for all vertices.
        
        all_vertices = []
        index_map = {}
        
        # Generate all points (side 0)
        for x1 in range(q):
            for x2 in range(q):
                # ... this is too slow to generate all combinations in Python loops for large d
                # But for d=5, q=4, it's 4^5 = 1024 points.
                pass
        
        # More efficient: iterate through all possible tuples
        # For q=4, d=5, we have 6 coordinates (x1..x6).
        # Let's just generate the connected component via BFS as before, but faster.
        
        start_vertex = (0, tuple([0] * (d + 1)))
        index, order, edges = {start_vertex: 0}, [start_vertex], []
        
        def get_neighbors(vertex):
            side, x = vertex
            neighbors = []
            # x is a tuple of length d+1, x[0] is ignored (or 0)
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_a[c], rule_b[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                neighbors.append((1 - side, tuple(y)))
            return neighbors

        for vertex in order:
            if time.time() > deadline - 20:
                break
            neighbors = get_neighbors(vertex)
            for other in neighbors:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def verify_girth(n, edges, k, g):
        """Verify the graph is k-regular, simple, and has girth >= g."""
        adj = [[] for _ in range(n)]
        edge_set = set()
        for i, j in edges:
            if i == j:
                return False
            if i > j:
                i, j = j, i
            if (i, j) in edge_set:
                return False
            edge_set.add((i, j))
            adj[i].append(j)
            adj[j].append(i)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Non-backtracking BFS from each vertex
        for start in range(n):
            visited = set()
            queue = [(start, -1, 1)]
            visited.add((start, -1))
            while queue:
                v, prev, dist = queue.pop(0)
                if dist >= g:
                    break
                for u in adj[v]:
                    if u == prev:
                        continue
                    if (u, v) in visited:
                        continue
                    if u == start and dist + 1 < g:
                        return False
                    visited.add((u, v))
                    if dist + 1 < g:
                        queue.append((u, v, dist + 1))
        return True

    # Try multiple d values and pick the smallest valid graph
    best_n = None
    best_edges = None

    if g <= 5:
        d_min = 3
    else:
        d_min = 3
        while True:
            girth_est = d_min + 5 if d_min % 2 == 1 else d_min + 4
            if girth_est >= g:
                break
            d_min += 1

    for d_try in range(d_min, d_min + 4):
        if time.time() > deadline - 30:
            break
        try:
            n_try, edges_try = luw_construct(k, g, d_try)
            if n_try < 50000:
                if verify_girth(n_try, edges_try, k, g):
                    if best_n is None or n_try < best_n:
                        best_n = n_try
                        best_edges = edges_try
        except Exception:
            pass

    if best_n is not None:
        return best_n, best_edges

    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
