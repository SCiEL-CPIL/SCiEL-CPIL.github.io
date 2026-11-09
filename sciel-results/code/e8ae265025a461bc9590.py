"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 135  # safety margin before 150s limit
    random.seed(42)

    # Moore bound lower bound: for k-regular girth g, n >= 2 * sum_{i=0}^{(g-1)//2} (k-1)^i for even g
    # or n >= 1 + k * sum_{i=0}^{(g-2)//2} (k-1)^i for odd g
    if g % 2 == 0:
        half = g // 2
        moore_lb = 2 * sum((k - 1)**i for i in range(half))
    else:
        half = (g - 1) // 2
        moore_lb = 1 + k * sum((k - 1)**i for i in range(half))

    def check_girth_fast(n, edges, min_girth):
        """Fast girth check using BFS with non-backtracking constraint."""
        if n == 0 or min_girth <= 2:
            return True

        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # For each vertex, do BFS but track previous vertex to avoid backtracking
        # A cycle of length < min_girth exists if we can return to start in < min_girth steps
        # without backtracking
        
        # Use BFS from each vertex, tracking (current, prev) pairs
        # Distance from start via non-backtracking walks
        
        # More efficient: for each edge (u,v), check if there's a non-backtracking path from v to u of length < min_girth-1
        # That would form a cycle of length < min_girth
        
        # Actually, let's use the standard approach: BFS from each vertex, but we need to track
        # the previous vertex to prevent backtracking. This is equivalent to BFS on the directed line graph.
        
        # For small graphs, let's just do BFS from each vertex with state (vertex, prev_vertex)
        # and check if we can reach (start, any) in less than min_girth steps
        
        # Simpler: check if there's any cycle of length < min_girth
        # A cycle of length L corresponds to a closed non-backtracking walk of length L
        # We need to check no such walk exists for L < min_girth
        
        # Use the fact that in a k-regular graph, the number of non-backtracking walks of length l
        # from a vertex can be computed, but we need to detect if any closes.
        
        # Practical approach: for each vertex s, do BFS on states (v, p) where v is current, p is previous
        # Start: for each neighbor nb of s, state (nb, s) at depth 1
        # If we reach state (s, nb) at depth d < min_girth, we found a cycle of length d+1... 
        # Wait, let me think more carefully.
        
        # A cycle s-v1-v2-...-vL-s has length L+1 where L is the number of edges from s back to s
        # via non-backtracking path. So if from s we go to v1 (depth 1), then v2 (depth 2), ..., 
        # and return to s at depth d, the cycle length is d.
        # We need d >= min_girth for all such returns.
        
        # So: BFS from each s, states are (current, prev). Start states: (nb, s) for each nb in adj[s], depth 1.
        # If we ever reach state (s, prev) at depth d where d < min_girth, there's a short cycle.
        # But we must ensure the walk is non-backtracking, which is enforced by not going back to prev.
        
        for s in range(n):
            if time.time() > deadline:
                return True
            # visited[v][prev] = depth
            # To save memory, use a dict or 2D array
            # n can be up to ~1000, so n*n = 1e6 which is fine
            visited = [[-1] * n for _ in range(n)]
            # visited[prev][v] = depth, or visited[v][prev]
            # Let's use visited[v][prev] = depth
            
            frontier = []
            for nb in adj[s]:
                visited[nb][s] = 1
                frontier.append((nb, s))
            
            for depth in range(1, min_girth):
                if time.time() > deadline:
                    return True
                new_frontier = []
                for v, prev in frontier:
                    for u in adj[v]:
                        if u == prev:
                            continue
                        if visited[u][v] == -1:
                            visited[u][v] = depth + 1
                            if u == s and depth + 1 < min_girth:
                                return False
                            new_frontier.append((u, v))
                frontier = new_frontier
                if not frontier:
                    break
        return True

    def voltage_graph_construct(k, g, deadline):
        """Construct a k-regular graph using voltage graph over small abelian group."""
        # For a k-regular graph, we can use a base graph with k edges per vertex
        # and assign voltages from a group to lift to a larger graph
        
        # Strategy: use a complete graph K_k as base (if k is small enough)
        # or a cycle C_k, and assign voltages from Z_m
        
        # For k=5, let's try base graph = C_5 (cycle of length 5)
        # Each vertex in C_5 has degree 2, so we need 5-regular... 
        # Actually, for a voltage graph, the base graph should be k-regular too,
        # or we use multiple edges.
        
        # Better: use a base graph that is k-regular. For k=5, we can use:
        # - The 5-cycle with chords? No, that changes the structure.
        # - Use a multigraph: take C_5 and add 3 more edges between each pair of adjacent vertices
        #   to make it 5-regular (each vertex connected to 2 neighbors with multiplicity 2.5... no)
        
        # Let's think differently. A voltage graph over Z_m with base graph B:
        # If B has n_b vertices and each vertex has degree d_b, then the lifted graph
        # has n_b * m vertices and is d_b-regular.
        # So we need d_b = k = 5.
        
        # Base graph options for 5-regular:
        # - K_6 (complete graph on 6 vertices) is 5-regular. n_b = 6, edges = 15.
        # - Or a 5-regular graph on fewer vertices... K_6 is the smallest 5-regular simple graph.
        
        # Let's use K_6 as base graph with voltage assignments in Z_m.
        # The lifted graph will have 6*m vertices and be 5-regular.
        
        # For girth >= g, we need to choose voltages such that no short cycles exist.
        
        # K_6 has triangles, so the lifted graph will have cycles corresponding to
        # voltage sums around triangles being 0 mod m.
        
        # Let's try different m values and optimize voltages.
        
        best_n = float('inf')
        best_edges = None
        
        # Base graph: K_6
        n_base = 6
        base_edges = [(i, j) for i in range(n_base) for j in range(i+1, n_base)]
        
        # Try different group sizes
        for m in range(3, 30):
            if time.time() > deadline - 5:
                break
            
            n = n_base * m
            if n < moore_lb:
                continue
            
            # Initialize voltages randomly
            # Each edge in base graph gets a voltage in Z_m
            # For undirected graph, voltage of (i,j) = -voltage of (j,i) mod m
            # So we assign voltage to each directed edge, with v(i,j) = -v(j,i)
            
            # Simulated annealing to maximize girth
            best_girth_found = 0
            best_voltages = None
            
            for sa_iter in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Initialize or perturb voltages
                if sa_iter == 0:
                    voltages = {}
                    for i, j in base_edges:
                        voltages[(i, j)] = random.randint(0, m - 1)
                        voltages[(j, i)] = (-voltages[(i, j)]) % m
                else:
                    # Perturb one random edge
                    i, j = random.choice(base_edges)
                    voltages[(i, j)] = random.randint(0, m - 1)
                    voltages[(j, i)] = (-voltages[(i, j)]) % m
                
                # Compute girth of lifted graph
                # Build adjacency of lifted graph
                # Vertex (v, a) in lifted graph connects to (w, a + voltages[(v,w)]) mod m
                # for each edge (v,w) in base graph
                
                adj = [[] for _ in range(n)]
                for v in range(n_base):
                    for w in range(n_base):
                        if v != w:
                            # Check if (v,w) is an edge in base graph
                            # K_6: all pairs are edges
                            volt = voltages[(v, w)]
                            for a in range(m):
                                u = w * m + (a + volt) % m
                                adj[v * m + a].append(u)
                
                # Check girth
                # Use the fast girth checker
                # But building edges list is expensive, let's check directly
                
                # Check for cycles of length < g
                # In the lifted graph, a cycle corresponds to a closed walk in the base graph
                # where the sum of voltages is 0 mod m.
                
                # For each cycle in the base graph, check if any voltage assignment
                # gives sum 0 mod m.
                
                # K_6 has many cycles. Let's just do BFS on the lifted graph.
                # But n = 6*m which can be up to 180, so it's manageable.
                
                # Build edges list for the checker
                edges = []
                seen_edges = set()
                for v in range(n_base):
                    for w in range(n_base):
                        if v < w:  # Only add each undirected edge once
                            volt = voltages[(v, w)]
                            for a in range(m):
                                u1 = v * m + a
                                u2 = w * m + (a + volt) % m
                                if u1 < u2:
                                    edges.append((u1, u2))
                                else:
                                    edges.append((u2, u1))
                
                # Check girth
                if check_girth_fast(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        best_voltages = {key: val for key, val in voltages.items()}
                        break  # Found a valid graph for this m, move to smaller m
                else:
                    # Try to improve
                    pass
        
        return best_n, best_edges

    # Use LUW construction as primary method
    q = k

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def get_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    def build_luw(d):
        rule = get_rule(d)

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

    # For LUW: girth >= d + 5 for odd d
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1

    best_n = float('inf')
    best_edges = None

    max_d = min(min_d + 4, g + 4)

    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 20:
            break
        try:
            n, edges = build_luw(d)
            if n < best_n:
                if check_girth_fast(n, edges, g):
                    best_n = n
                    best_edges = edges
        except Exception:
            continue

    # Try voltage graph construction if we have time
    if time.time() < deadline - 30:
        try:
            vn, ve = voltage_graph_construct(k, g, deadline - 10)
            if vn < best_n:
                best_n = vn
                best_edges = ve
        except Exception:
            pass

    if best_edges is None:
        d = min_d
        n, edges = build_luw(d)
        return n, edges

    return best_n, best_edges
# EVOLVE-BLOCK-END
