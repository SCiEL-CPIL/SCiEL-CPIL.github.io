import time
import random

def construct(k, g):
    """
    Construct a k-regular graph of girth g with as few vertices as possible.
    Strategy:
    1. Try Random Voltage Lifts of small base graphs (Petersen, etc.) for k=3.
       This is a structural change from algebraic constructions (LUW/Ramanujan)
       which often produce large graphs or fail to optimize vertex count in time.
       Voltage lifts allow fine-tuning of cycle lengths by choosing voltages.
    2. Fallback to Lazebnik-Ustimenko-Woldar (LUW) construction if lifts fail
       or time runs out, as it provides a guaranteed valid (though potentially
       large) graph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    def remaining():
        return deadline - time.time()

    # ---------------- Girth Verification ----------------
    def check_girth(n, edges, g, time_limit):
        """
        Returns the girth (length of shortest cycle) if < time_limit,
        or -1 if timeout.
        Uses BFS from each vertex to find shortest cycle.
        """
        if n == 0:
            return g
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Quick check for degree
        for i in range(n):
            if len(adj[i]) != k:
                return -1  # Invalid graph

        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Base Graph Definitions ----------------
    def get_petersen_edges():
        """
        Petersen graph: 10 vertices, 3-regular, girth 5.
        Outer cycle: 0-1-2-3-4-0
        Inner star: 5-7-9-6-8-5
        Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        """
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return edges

    def get_heawood_edges():
        """
        Heawood graph: 14 vertices, 3-regular, girth 6.
        Kneser graph KG(7,2).
        Vertices are pairs (i,j) with i<j from 0..6.
        Edge if disjoint.
        """
        vertices = []
        for i in range(7):
            for j in range(i+1, 7):
                vertices.append((i, j))
        n = len(vertices)
        idx = {v: i for i, v in enumerate(vertices)}
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if len(set(vertices[i]) & set(vertices[j])) == 0:
                    edges.append((i, j))
        return edges

    def get_random_3_reg(n):
        """
        Generate a random 3-regular graph on n vertices (n must be even).
        Uses configuration model with simple swap repair.
        """
        if n < 4 or n % 2 != 0:
            return None
        
        # Stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        # Pair stubs
        edges = set()
        adj = {i: [] for i in range(n)}
        
        # Initial pairing
        for i in range(0, n, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Loop
            if (min(u,v), max(u,v)) in edges:
                return None # Multi-edge
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
            
        return list(edges)

    # ---------------- Voltage Lift Construction ----------------
    def build_voltage_lift(base_edges, m):
        """
        Build a voltage graph lift of base_edges over Z_m.
        Assigns random voltages to edges.
        Returns (n, edges) or None.
        """
        n_base = max([u for e in base_edges for u in e] + [v for e in base_edges for v in e]) + 1
        n = n_base * m
        
        # Map base vertex v to fiber {v*m, v*m+1, ..., v*m+m-1}
        # We need to assign voltages to each edge.
        # To ensure 3-regularity and connectivity, we use random voltages.
        # However, simply random voltages might result in disconnected components
        # or small cycles.
        
        # A better strategy for voltage lifts:
        # If we assign voltage 0 to all edges, we get m disjoint copies.
        # We want a connected lift.
        
        # Let's try a specific heuristic:
        # Assign random voltages in 1..m-1 to ensure mixing.
        
        lift_edges = []
        # We need to be careful: the lift of an edge (u,v) with voltage w
        # connects (u, z) to (v, z+w) for all z in Z_m.
        
        # To make it simple and fast, we'll just generate the edges.
        # Since we are checking girth afterwards, we can afford to generate
        # a potentially invalid (disconnected) graph, but the check will catch it.
        # However, the prompt requires "every vertex must have degree exactly k".
        # Voltage lifts of k-regular graphs are k-regular.
        # The main risk is disconnectedness or small girth.
        
        # Optimization: Only try lifts that are likely to be connected.
        # A random voltage assignment on a connected base graph usually yields
        # a connected lift if m is large enough, but not guaranteed.
        
        # Let's just generate and check.
        
        # To speed up, we can pre-compute the structure.
        # For each edge (u,v) in base:
        #   pick a random voltage w in 0..m-1
        #   add edges (u*m + z, v*m + (z+w)%m) for z in 0..m-1
        
        # This is O(E_base * m) edges.
        # For m=100, E_base=15, this is 1500 edges. Fast.
        
        # We need to ensure no self-loops or multi-edges in the lift.
        # If base is simple, lift is simple if voltages are consistent?
        # Actually, if base has no loops/multi-edges, lift is simple.
        
        # Let's generate.
        # We use a deterministic seed based on m and base to allow reproducibility?
        # No, random is better for search. But we need to be fast.
        
        # We will try a few random voltage assignments.
        
        return None # Placeholder, we will implement in main loop

    def try_voltage_lift(base_edges, m, attempts=5):
        """
        Try to find a valid voltage lift of base_edges over Z_m.
        """
        n_base = max([u for e in base_edges for u in e] + [v for e in base_edges for v in e]) + 1
        n = n_base * m
        
        for _ in range(attempts):
            # Assign random voltages
            voltages = {}
            for u, v in base_edges:
                # Random voltage in 1..m-1 to avoid trivial components
                # 0 is also allowed but less mixing
                voltages[(u, v)] = random.randint(0, m-1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            
            edges = []
            # Build lift
            for u, v in base_edges:
                w = voltages[(u, v)]
                # For each z in Z_m
                # Edge from (u, z) to (v, z+w)
                u_base = u
                v_base = v
                for z in range(m):
                    u_node = u_base * m + z
                    v_node = v_base * m + (z + w) % m
                    edges.append((u_node, v_node))
            
            # Check validity
            # 1. Degree check
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != k for d in deg):
                continue
            
            # 2. Simple graph check (no multi-edges)
            # Since base is simple, and we map (u,z)->(v,z+w),
            # can we have two edges between same nodes?
            # Only if there are two edges between u,v in base, which there aren't.
            # So it is simple.
            
            # 3. Girth check
            # We don't know girth yet. We will check in main loop.
            # But we can do a quick check for small cycles?
            # No, just return the candidate.
            
            return n, edges
            
        return None

    # ---------------- LUW Fallback ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        else:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for variant in range(2):
            if remaining() < 15:
                break
            rule = generate_rules(d, variant)
            res = build_luw(q, d, rule, deadline - 15)
            if res is None:
                continue
            n, edges = res
            if n >= best_n:
                continue
            gc = check_girth(n, edges, g, deadline - 10)
            if gc == -1:
                continue
            if gc >= g:
                best, best_n = (n, edges), n

        if best is None:
            # Guaranteed fallback
            rule = generate_rules(d, 0)
            res = build_luw(q, d, rule, deadline - 5)
            if res is not None:
                best = res
        return best

    # ---------------- Main Search Logic ----------------
    
    best_result = None
    best_n = float('inf')

    # 1. Try Voltage Lifts
    # This is the structural change.
    # We target specific vertex counts by choosing m.
    # Moore bound for k=3, g=16 is 512.
    # Petersen lift: n = 10 * m.
    # Heawood lift: n = 14 * m.
    
    base_graphs = [
        ("petersen", get_petersen_edges()),
        ("heawood", get_heawood_edges())
    ]
    
    # We want n close to 512.
    # For Petersen (10 vertices): m=50 -> 500, m=52 -> 520
    # For Heawood (14 vertices): m=36 -> 504, m=37 -> 518
    
    # We will try a range of m values.
    # Time budget for lifts: 30 seconds.
    lift_deadline = start_time + 30.0
    
    # Pre-compute base edges
    base_edges_list = []
    for name, edges in base_graphs:
        base_edges_list.append(edges)
        
    # Determine m ranges to try
    # We want n between Moore Bound and some upper limit.
    # Let's try m from 20 to 60.
    
    for base_idx, base_edges in enumerate(base_edges_list):
        n_base = max([u for e in base_edges for u in e] + [v for e in base_edges for v in e]) + 1
        if n_base == 0: continue
        
        # Try m values
        # We prioritize m that yields n near 512.
        target_n = 512
        m_target = target_n // n_base
        
        # Try m around m_target
        m_range = []
        for m in range(max(2, m_target - 10), m_target + 11):
            m_range.append(m)
        
        # Shuffle to avoid bias? Or just iterate.
        # We'll iterate but check time.
        
        for m in m_range:
            if time.time() > lift_deadline:
                break
            
            n, edges = try_voltage_lift(base_edges, m, attempts=3)
            if n is None:
                continue
            
            # Check girth
            # We need to be careful with time here.
            # Girth check is O(n^2) roughly? No, O(n * (n-1)) worst case.
            # For n=500, this is 250k operations. Fast.
            
            # We only check if n is potentially better than current best
            if n < best_n:
                gc = check_girth(n, edges, g, time.time() + 5.0) # 5s timeout per check
                if gc != -1 and gc >= g:
                    best_result = (n, edges)
                    best_n = n
                    # If we found a very good one, we can break?
                    # No, keep searching for smaller n.
                    # But if n is close to Moore bound, it's likely optimal.
                    if n <= 520:
                        # Found a very strong candidate
                        pass

    # 2. Fallback to LUW
    if remaining() > 10:
        luw_res = luw_construct(k, g)
        if luw_res:
            n, edges = luw_res
            if n < best_n:
                # Verify LUW girth just in case
                gc = check_girth(n, edges, g, deadline - 2)
                if gc != -1 and gc >= g:
                    best_result = (n, edges)
                    best_n = n

    # Final Fallback: If no valid graph found (shouldn't happen with LUW)
    if best_result is None:
        # Force a LUW construction with relaxed time check
        q = k
        d = max(3, (g - 2) // 2)
        rule = generate_rules(d, 0)
        res = build_luw(q, d, rule, deadline - 1)
        if res:
            best_result = res
        else:
            # Absolute last resort: Return a trivial valid graph?
            # No, must be k-regular.
            # If k=3, n must be even.
            # A cycle of length n is 2-regular.
            # We need 3-regular.
            # Just return the LUW graph even if time ran out?
            # The prompt says "invalid result... scores 0".
            # So we MUST return a valid graph.
            # LUW construction is deterministic and fast enough for small d.
            # If it timed out, something is wrong.
            # Let's try a smaller d? No, girth must be >= g.
            # We will return the LUW graph generated with a very tight deadline.
            res = build_luw(q, d, rule, deadline - 0.1)
            best_result = res

    return best_result

# Test harness (optional, for local debugging)
if __name__ == "__main__":
    # k=3, g=16
    # n, edges = construct(3, 16)
    # print(f"n={n}, edges={len(edges)}")
    pass
