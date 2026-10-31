import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback: The seed LUW graph
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

    # Get fallback result immediately to have a valid answer
    fallback_n, fallback_edges = luw_construct(k, g)
    best_n = fallback_n
    best_edges = fallback_edges
    
    # Strategy: Voltage Graph Lifting
    # Base graph: Tutte-Coxeter Graph (30 vertices, 3-regular, girth 8)
    # Vertices are (side, i, j) where side in {0,1}, i in 0..2, j in 0..3? 
    # Actually, let's use a known adjacency list for Tutte-Coxeter.
    # It is a bipartite graph with 15+15 vertices.
    # Let's generate it or hardcode a small 3-regular base.
    # For simplicity and robustness, let's use the **Heawood Graph** (14 vertices, girth 6) 
    # or **Petersen** (10 vertices, girth 5).
    # Tutte-Coxeter is better. Let's construct it.
    # Tutte-Coxeter is the incidence graph of the generalized quadrangle GQ(2,2) ? No, that's Heawood.
    # Tutte-Coxeter is the incidence graph of the generalized quadrangle GQ(2,2) with 15 points and 15 lines? 
    # GQ(2,2) has (s+1)(st+1) = 3*5=15 points. Yes.
    # So 30 vertices total.
    
    # Generate GQ(2,2) incidence graph
    # Points: (x, y, z) in GF(3)^3 not all 0? No, GQ(2,2) is over GF(3)?
    # Actually, let's just use a hardcoded adjacency for a 3-regular graph with girth 8.
    # Or simpler: Use the **Möbius-Kantor** graph (16 vertices, girth 8).
    # It is a Cayley graph of Z8 x Z2? Or Z16?
    # Let's use a generic search on a base graph.
    
    # Base Graph: The 16-vertex Möbius-Kantor graph
    # Vertices 0..15
    # Edges:
    mk_edges = []
    # It is a cubic graph.
    # Let's define it by generators in Z8 x Z2?
    # Or just use the Heawood graph (14 vertices) as it's smaller and easier to verify.
    # Heawood Graph: 14 vertices, girth 6.
    # If we lift Heawood with m, n = 14*m.
    # If m=64, n=896.
    # If we can get girth 16 with m=32, n=448.
    
    # Let's try to lift the Heawood graph.
    # Heawood Graph Adjacency:
    # It is the incidence graph of Fano plane.
    # Points 0..6, Lines 7..13.
    # Fano plane lines:
    fano_lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    heawood_edges = []
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    
    n_base = 14
    base_adj = [[] for _ in range(n_base)]
    for u, v in heawood_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Ensure 3-regular
    for i in range(n_base):
        if len(base_adj[i]) != 3:
            # Fallback if construction failed
            return best_n, best_edges

    # Search for voltage assignments
    # Voltage graph: Vertices (v, i) where v in 0..13, i in 0..m-1
    # Edge between (u, i) and (v, j) exists if (u,v) in base and j = (i + voltage(u,v)) % m
    # We need to assign voltage to each directed edge? 
    # For undirected base edge {u,v}, we assign a voltage w.
    # Then (u, i) connects to (v, (i+w)%m) and (v, i) connects to (u, (i-w)%m).
    # This preserves 3-regularity.
    
    # We will try different m values.
    # Lower bound for m: If base girth is 6, and we want girth 16.
    # Random twists usually increase girth.
    
    def check_girth_lifted(m, voltages):
        # voltages is a dict (u, v) -> w for u < v
        # Build adjacency for lifted graph
        n_lifted = n_base * m
        adj = [[] for _ in range(n_lifted)]
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v:
                    w = voltages.get((u, v), 0)
                    # u -> v
                    for i in range(m):
                        ui = u * m + i
                        vj = v * m + (i + w) % m
                        adj[ui].append(vj)
                    # v -> u
                    for j in range(m):
                        vj = v * m + j
                        ui = u * m + (j - w) % m
                        adj[vj].append(ui)
                elif u > v:
                    # Already handled when v < u
                    pass
        # Check girth exactly using BFS from each vertex? 
        # Too slow for n=14*64=896 if done naively for every candidate.
        # We can check for cycles of length < g.
        # Girth is the shortest cycle.
        # We can do BFS from each vertex up to depth (g-1)//2.
        # If we find a vertex at depth d that is also connected to a vertex at depth d' 
        # such that d+d' + 1 < g, we have a short cycle.
        # Actually, simpler: BFS from each vertex, keep parent. 
        # If we encounter a visited node, check path length.
        # To speed up, we can stop BFS if we find a cycle of length >= g? No, we need to know if ANY cycle is < g.
        # So we must find the shortest cycle.
        
        # Efficient girth check for sparse graphs:
        # For each edge (u, v), remove it and find shortest path between u and v.
        # Girth is min over all edges of (shortest_path(u, v, without edge) + 1).
        # This is O(E * (V+E)) which is ~ 1300 * 1000 = 1.3M ops per check.
        # If we do 1000 checks, 1.3B ops. Might be too slow in Python.
        
        # Alternative: BFS from each vertex up to depth g/2.
        # If we find a back-edge to a vertex in the BFS tree at a level that creates a cycle < g, return False.
        # Max depth 7 for g=16.
        # 896 vertices * 3^7 ~ 896 * 2187 ~ 1.9M ops per check.
        # Still potentially slow for many trials.
        
        # Let's try a few trials.
        
        def bfs_shortest_cycle(u, v, exclude_edge):
            # Find shortest path from u to v not using exclude_edge
            # Since graph is small, simple BFS
            visited = {u: None}
            q = deque([u])
            while q:
                curr = q.popleft()
                if curr == v:
                    # Reconstruct path length
                    path_len = 0
                    node = v
                    while node != u:
                        node = visited[node]
                        path_len += 1
                    return path_len
                for nei in adj[curr]:
                    if (curr, nei) == exclude_edge or (nei, curr) == exclude_edge:
                        continue
                    if nei not in visited:
                        visited[nei] = curr
                        q.append(nei)
            return float('inf')

        min_girth = float('inf')
        # Sample a few edges to estimate girth? 
        # No, must be exact.
        # But we can break early if we find a cycle < g.
        
        # Optimization: Only check cycles that could be short.
        # In a 3-regular graph, if girth >= g, then number of vertices in ball of radius r is 1 + 3 * (3^(r-1)-1)/2.
        
        # Let's just do BFS from each vertex up to depth (g-1)//2.
        # If we find a visited vertex that is not the parent, and the cycle length is < g, return False.
        
        max_depth = (g - 1) // 2
        
        for start in range(n_lifted):
            # BFS
            dist = [-1] * n_lifted
            parent = [-1] * n_lifted
            dist[start] = 0
            q = deque([start])
            found_short = False
            while q:
                u = q.popleft()
                if dist[u] > max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] <= max_depth:
                            q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Back edge or cross edge
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            # If we finish BFS from start without finding short cycle, continue
        return True

    # Try different m
    # Heuristic: Start with small m, increase if girth fails.
    # But we want MINIMUM n. So try small m first.
    # m must be such that 14*m >= Moore bound? 
    # Moore bound for k=3, g=16 is 512.
    # So m >= 512/14 = 36.6. So m >= 37.
    # Let's try m in [37, 40, 45, 50, 64, 128]
    
    m_candidates = [37, 38, 39, 40, 42, 45, 48, 50, 56, 64, 128]
    
    for m in m_candidates:
        if time.time() > deadline:
            break
        if 14 * m >= best_n:
            break # No improvement possible
        
        # Try random voltage assignments
        num_trials = 50
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            voltages = {}
            for u, v in heawood_edges:
                # Random voltage in 1..m-1 (avoid 0 to increase girth?)
                # Actually 0 is fine, but non-zero might help break symmetry
                voltages[(u, v)] = random.randint(1, m - 1)
            
            if check_girth_lifted(m, voltages):
                # Construct edges
                n_lifted = 14 * m
                lifted_edges = []
                for u, v in heawood_edges:
                    w = voltages[(u, v)]
                    for i in range(m):
                        ui = u * m + i
                        vj = v * m + (i + w) % m
                        lifted_edges.append((ui, vj))
                
                if n_lifted < best_n:
                    best_n = n_lifted
                    best_edges = lifted_edges
                    
    return best_n, best_edges

# EVOLVE-BLOCK-END
