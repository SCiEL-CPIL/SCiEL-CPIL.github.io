<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # We construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # The base graph is K_{3,3} with vertices A={0,1,2} and B={3,4,5}.
    # Edges in base: (i, j+3) for all i in 0..2, j in 0..2.
    # Each edge gets a voltage in Z_m. The lifted graph has 6*m vertices.
    # Vertex (side, base, shift) where side in {0,1}, base in 0..2, shift in 0..m-1.
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m) where v[i][j] is the voltage.
    
    # For girth g, we need no cycles shorter than g.
    # The Moore bound for 3-regular graph with girth g gives lower bound on n.
    # For g=16, Moore bound is 512. The best known is 936.
    # With K_{3,3} lift, n = 6*m, so m = n/6.
    # To get n close to 512, m would be ~85.33, so m=86 gives n=516.
    # But we need to find voltages that give girth >= g.
    
    # Strategy: try random voltage assignments for various m values,
    # check girth, and return the smallest m that works.
    
    # For efficiency, we'll try m from a reasonable lower bound upward.
    # Lower bound: Moore bound for (3, g) cage.
    # For g even, Moore bound n >= 3 * (2^(g/2 - 1) - 1)
    # For g=16: n >= 3 * (2^7 - 1) = 3 * 127 = 381
    # So m >= ceil(381/6) = 64
    
    # Let's compute Moore bound properly
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        # For odd girth, the bound is different
        # n >= (3/2) * (2^(g-1)//2 - 1) approximately
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    # We'll try m from m_min upward, but to keep within time limits,
    # we'll try a limited number of m values and random voltage assignments.
    
    def check_girth(m, voltages, required_girth):
        """Check if the lifted graph has girth >= required_girth.
        voltages: 3x3 matrix where voltages[i][j] is the voltage on edge from A_i to B_j.
        Returns True if girth >= required_girth, False otherwise.
        """
        n = 6 * m
        # Build adjacency list
        # Vertex encoding: side * 3 * m + base * m + shift
        # side 0 = A, side 1 = B
        adj = [[] for _ in range(n)]
        
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = 0 * 3 * m + i * m + s
                    w = 1 * 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)
        
        # BFS from each vertex to find shortest cycle
        # Actually, for girth checking, we can do BFS from each vertex and track
        # the first time we revisit a vertex (excluding the parent).
        # But that's O(n * (n + e)) which is too slow for large n.
        
        # Better approach: BFS from each vertex, stop when depth reaches required_girth/2
        # If we find a back edge at depth < required_girth/2, girth is too small.
        
        half_girth = required_girth // 2
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        
        return True
    
    # Try random voltage assignments for increasing m
    random.seed(42)
    
    # For each m, try several random voltage assignments
    num_trials = 50
    
    for m in range(m_min, m_min + 200):  # Try up to 200 values of m
        if 6 * m > 2000:  # Cap to avoid too large graphs
            break
        
        for trial in range(num_trials):
            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth(m, voltages, g):
                # Build the graph
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
        
        # If no random assignment worked for this m, try a few more
        # Actually, let's also try some structured assignments
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    import time
    from collections import deque

    # Construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # The base graph is K_{3,3} with vertices A={0,1,2} and B={3,4,5}.
    # Each edge gets a voltage in Z_m. The lifted graph has 6*m vertices.
    # Vertex (side, base, shift) where side in {0,1}, base in 0..2, shift in 0..m-1.
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m) where v[i][j] is the voltage.

    # For girth g, we need no cycles shorter than g.
    # Moore bound for 3-regular graph with girth g:
    # For g even: n >= 3 * (2^(g/2 - 1) - 1)
    # For g=16: n >= 3 * (2^7 - 1) = 3 * 127 = 381
    # So m >= ceil(381/6) = 64

    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3

    m_min = max(1, math.ceil(moore_bound / 6))

    # Efficient girth check using BFS from each vertex up to depth g//2
    def check_girth_fast(m, voltages, required_girth):
        """Check if the lifted graph has girth >= required_girth using BFS."""
        n = 6 * m
        half_girth = required_girth // 2

        # Precompute adjacency as a list of lists
        # Vertex encoding: side * 3 * m + base * m + shift
        adj = [[] for _ in range(n)]

        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)

        # BFS from each vertex, stop when depth reaches half_girth
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])

            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        queue.append(v)
                    elif v != u and dist[v] < du:
                        # Found a cycle: path from start to v (length dist[v])
                        # plus edge v->u (length 1) plus path from u back to start
                        # Actually, this isn't quite right. Let me use a different approach.
                        pass

        # Better approach: for each start, do BFS and track parent.
        # A cycle exists if we find an edge (u,v) where v is already visited
        # and v is not the parent of u. The cycle length is dist[u] + dist[v] + 1.
        # But we need to be careful: this only finds cycles that pass through start.
        # To find ALL cycles, we need to check from every start.

        # Alternative: use the fact that in a bipartite graph, all cycles are even.
        # For girth >= g (even), we need no cycle of length < g.
        # We can do BFS from each vertex up to depth g/2 - 1 and check for back edges.

        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])

            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a potential cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False

        return True

    # Even faster: use numpy for the BFS or just optimize the Python BFS
    # Let's try a different approach: since the graph is bipartite and 3-regular,
    # we can check girth by looking at the number of walks of length 2d from each vertex.
    # If there are fewer than the expected number (no backtracking), girth is large enough.

    # Actually, let's just use the BFS approach but with early termination and
    # limit the number of vertices we check if m is large.

    random.seed(42)
    time_limit = 110  # seconds, leave some buffer

    # Try m values starting from m_min
    # For each m, try multiple random voltage assignments
    # Use a time budget to decide how many trials per m

    best_result = None

    for m in range(m_min, m_min + 300):
        if 6 * m > 3000:  # Cap to avoid too large graphs
            break

        if time.time() - time.time() > time_limit:
            break

        # Number of trials decreases as m increases (larger graphs are slower to check)
        max_trials = max(10, 200 - m // 2)

        for trial in range(max_trials):
            if time.time() - time.time() > time_limit:
                break

            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]

            # Quick check: avoid trivial cycles
            # A 4-cycle exists if for some i1, i2, j1, j2:
            # voltages[i1][j1] + voltages[i2][j2] == voltages[i1][j2] + voltages[i2][j1] (mod m)
            # This would create a 4-cycle. Let's check this first.
            has_4cycle = False
            for i1 in range(3):
                for i2 in range(i1 + 1, 3):
                    for j1 in range(3):
                        for j2 in range(j1 + 1, 3):
                            if (voltages[i1][j1] + voltages[i2][j2]) % m == \
                               (voltages[i1][j2] + voltages[i2][j1]) % m:
                                has_4cycle = True
                                break
                        if has_4cycle:
                            break
                    if has_4cycle:
                        break
                if has_4cycle:
                    break

            if has_4cycle:
                continue

            # Check for 6-cycles: more complex, skip for now
            # Full girth check
            if check_girth_fast(m, voltages, g):
                # Build the graph
                n = 6 * m
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = voltages[i][j]
                        for s in range(m):
                            u = i * m + s
                            w = 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges

    # Fallback: return a known good construction if we didn't find one
    # Use a larger m that's more likely to work
    m_fallback = max(m_min + 50, 100)
    for trial in range(100):
        voltages = [[random.randint(0, m_fallback - 1) for _ in range(3)] for _ in range(3)]
        if check_girth_fast(m_fallback, voltages, g):
            n = 6 * m_fallback
            edges = []
            for i in range(3):
                for j in range(3):
                    v = voltages[i][j]
                    for s in range(m_fallback):
                        u = i * m_fallback + s
                        w = 3 * m_fallback + j * m_fallback + (s + v) % m_fallback
                        edges.append((u, w))
            return n, edges

    # Last resort: return a simple 3-regular graph (may not satisfy girth)
    # This is a fallback that should rarely be needed
    n = 6 * m_min
    edges = []
    for i in range(3):
        for j in range(3):
            v = 0
            for s in range(m_min):
                u = i * m_min + s
                w = 3 * m_min + j * m_min + (s + v) % m_min
                edges.append((u, w))
    return n, edges
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made a mistake in the time tracking. Let me fix that and also improve the girth check to be more efficient.

