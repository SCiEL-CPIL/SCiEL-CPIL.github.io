import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit

    # Helper: Check if a graph (n, adj) has girth at least g
    # Returns True if girth >= g, False otherwise
    def check_girth_valid(n, adj, min_girth):
        # For each vertex, BFS to find shortest cycle
        # In a simple graph, a cycle of length L exists iff there is a back edge 
        # in BFS tree connecting u and v where dist[u] + dist[v] + 1 = L 
        # and the paths are disjoint except at start.
        # Actually, simpler: 
        # A cycle of length L exists if there is a path of length L-1 from v to w 
        # such that (v,w) is an edge and w is not the parent of v in the BFS tree rooted at v?
        # No, standard check:
        # For each start node s, run BFS.
        # If we find a neighbor v of u such that v is not parent of u and u is not parent of v,
        # then we have a cycle. The length is dist[u] + dist[v] + 1.
        # We must ensure the paths from s to u and s to v are disjoint except at s.
        # In BFS, this is true if u and v are in different branches or if one is not ancestor of other.
        # Since we check dist[u] + dist[v] + 1, if u is ancestor of v, dist[v] = dist[u] + 1, 
        # then cycle length would be 2*dist[u] + 2, but the edge (u,v) is the tree edge, which we skip.
        # If u is not ancestor of v, the paths are disjoint except at s.
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        # Check if paths are disjoint
                        # If u is ancestor of v or v is ancestor of u, they share path.
                        # In BFS, if dist[u] == dist[v], they are not ancestors.
                        # If dist[u] < dist[v], u could be ancestor.
                        # We can just check the length. If the length is < min_girth, return False.
                        # The length is dist[u] + dist[v] + 1.
                        # Is this always the length of the shortest cycle through s?
                        # Yes, for the specific cycle formed by s->...->u - v <-...<-s.
                        # If this cycle is short, the girth is short.
                        cyc_len = dist[u] + dist[v] + 1
                        if cyc_len < min_girth:
                            return False
        return True

    # Helper: Convert edge list to adjacency list
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Remove duplicates if any (should be simple)
        for i in range(n):
            adj[i] = list(set(adj[i]))
        return adj

    # Candidate 1: Lazebnik-Ustimenko-Woldar
    def luw_construct(k, g):
        q = k
        # LUW graphs are defined for q >= 3.
        # d is related to girth.
        # For girth g, we need d such that the graph has no cycles < g.
        # LUW(q, d) has girth 2*d + 1 or 2*d + 2 depending on parameters.
        # Actually, LUW(q, d) is a (q-1)-regular graph? No, k=q in this context usually means degree q-1?
        # The seed code uses q=k. Let's stick to the seed's logic but verify.
        # Seed: q=k. neighbours yields q neighbors?
        # In seed: for first in range(q): ... yield ...
        # So degree is q.
        # Girth of LUW(q, d) is at least 2*d - 1?
        # Let's just use the seed's parameter mapping and verify girth.
        
        # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # This is a heuristic. We should try a few d values.
        best = None
        for d in range(3, g + 5):
            if time.time() > deadline:
                break
            try:
                n, edges = _luw_impl(k, d)
                if n == 0: continue
                adj = build_adj(n, edges)
                # Check degree
                if any(len(adj[i]) != k for i in range(n)):
                    continue
                # Check girth
                if check_girth_valid(n, adj, g):
                    if best is None or n < best[0]:
                        best = (n, edges)
            except Exception:
                pass
        return best

    def _luw_impl(q, d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule generation from seed
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

    # Candidate 2: Generalized Hexagon of order (4,4)
    # This is a 5-regular graph with girth 12 and 330 vertices.
    def gh4_construct():
        # Incidence graph of generalized hexagon H(4).
        # Vertices are points and lines of PG(2,4)? No, it's the incidence graph of the generalized hexagon.
        # Order (s,t) = (4,4).
        # Number of points = 1 + 4 + 4^2 + ... + 4^5 = 1365? No.
        # Formula for number of points in generalized hexagon of order (s,t):
        # N = 1 + s + st + st^2 + st^3 + st^4 = 1 + 4 + 16 + 64 + 256 + 1024 = 1365?
        # Wait, the (4,4) generalized hexagon has 330 vertices?
        # Let's check known cages.
        # The (5,12) cage has 330 vertices? 
        # Exoo et al. 2026 paper mentions (5,9) is 1116.
        # What is the (5,12) cage?
        # The incidence graph of the generalized hexagon of order (4,4) is a (5,12) cage?
        # Actually, the generalized hexagon of order (4,4) has parameters (s,t)=(4,4).
        # Number of points = 1 + s(1+t)(1+t^2)(1+t^3)(1+t^4)? No.
        # Let's look up "5-regular graph girth 12".
        # The Hoffman-Singleton graph is (7,5) with 50 vertices.
        # The (5,12) cage is not widely known as a single small graph in standard tables like (3,9) or (7,5).
        # However, the incidence graph of the generalized quadrangle Q(4,4) is (5,8) with 112 vertices?
        # Let's skip GH4 if I can't verify the exact vertex count and structure easily without error.
        # I will rely on LUW for k=5 unless I find a specific known smaller cage.
        # Actually, for k=5, g=9, the best known is 1116. LUW might produce something close or better.
        # For k=5, g=12, is there a known cage?
        # Let's just return None for specific hard cases and rely on LUW.
        return None

    # Candidate 3: Hoffman-Singleton (7,5)
    def hs_construct():
        # 50 vertices, 7-regular, girth 5.
        # Circulant C_50(1, 7, 11)? No, HS is not a circulant.
        # It can be constructed using the Petersen graph.
        # Vertices: pairs (i, j) where i in 0..9, j in 0..4? No.
        # Construction:
        # Take 5 copies of the Petersen graph? No.
        # Standard construction:
        # Vertices are {0..9} x {0..4}. Total 50.
        # Edges:
        # 1. Within each fiber {i} x {0..4}, connect j to (j+1)%5 and (j-1)%5? No, that's degree 2.
        # The Hoffman-Singleton graph is a (7,5) cage.
        # It is a circulant? No.
        # It is a "doubly regular" graph.
        # Let's use a known edge list or a generator.
        # Generator:
        # Let V = {0, ..., 49}.
        # Partition into 5 sets of 10?
        # Actually, there is a simple description:
        # It is the complement of a 5-regular graph? No.
        # Let's use the known fact that it is a circulant graph C_50(1, 7, 11)?
        # Let's check C_50(1, 7, 11).
        # Degree 6? 1, -1, 7, -7, 11, -11. That's 6 neighbors. We need 7.
        # C_50(1, 7, 11, ...)?
        # Actually, HS is not a circulant.
        # I will skip explicit HS construction to avoid bugs, unless I can verify it quickly.
        # LUW for k=7, g=5 might work.
        return None

    # Candidate 4: (3,9) Cage (46 vertices)
    def c39_construct():
        # The (3,9) cage is 46 vertices.
        # It is a circulant?
        # C_46(1, 10, 16)?
        # Let's test C_46(1, 10, 16).
        n = 46
        gen = [1, 10, 16]
        edges = []
        for u in range(n):
            for v in gen:
                edges.append((u, (u + v) % n))
        # Deduplicate
        edges = list(set((min(u,v), max(u,v)) for u,v in edges))
        adj = build_adj(n, edges)
        # Check degree
        if any(len(adj[i]) != 3 for i in range(n)):
            return None
        # Check girth
        if check_girth_valid(n, adj, 9):
            return n, edges
        return None

    # Candidate 5: Petersen (3,5)
    def petersen_construct():
        n = 10
        edges = []
        # Outer pentagon
        for i in range(5):
            edges.append((i, (i+1)%5))
        # Inner star
        for i in range(5):
            edges.append((5+i, 5+(i+2)%5))
        # Spokes
        for i in range(5):
            edges.append((i, 5+i))
        adj = build_adj(n, edges)
        if check_girth_valid(n, adj, 5):
            return n, edges
        return None

    # Candidate 6: Heawood (3,6)
    def heawood_construct():
        n = 14
        # Incidence graph of Fano plane
        # 7 points, 7 lines.
        # Points 0-6, Lines 7-13.
        edges = []
        # Fano plane lines:
        # 012, 034, 056, 135, 146, 236, 245
        lines = [
            (0,1,2), (0,3,4), (0,5,6),
            (1,3,5), (1,4,6), (2,3,6), (2,4,5)
        ]
        for i, line in enumerate(lines):
            line_id = 7 + i
            for p in line:
                edges.append((p, line_id))
        adj = build_adj(n, edges)
        if check_girth_valid(n, adj, 6):
            return n, edges
        return None

    # Candidate 7: Pappus (3,8) - Wait, Pappus is (3,6)? No, Pappus graph is (3,8)?
    # Pappus graph has 18 vertices, 3-regular, girth 8?
    # Actually, the (3,8) cage is the Möbius-Kantor graph (16 vertices).
    # The (3,7) cage is the Pappus graph (18 vertices)? No, (3,7) is 18 vertices?
    # Let's check:
    # g=5: 10 (Petersen)
    # g=6: 14 (Heawood)
    # g=7: 18 (Pappus? No, Pappus is girth 8? Or 7?)
    # Pappus graph: 18 vertices, 3-regular. Girth 8?
    # Actually, the (3,7) cage is 18 vertices.
    # The (3,8) cage is 16 vertices (Möbius-Kantor).
    # Let's implement Möbius-Kantor for (3,8).
    def mobius_kantor_construct():
        n = 16
        # Circulant C_16(1, 5)?
        # C_16(1, 5) has degree 4? No, 1, -1, 5, -5. Degree 4.
        # We need degree 3.
        # C_16(1, 5, ...)?
        # Möbius-Kantor is a circulant?
        # It is a "cubic" graph.
        # Let's try to find a generator set for n=16, k=3.
        # 3 is odd, so it can't be a standard circulant with symmetric generators unless one is n/2?
        # If n is even, and k is odd, one generator must be n/2.
        # So gen = {a, b, 8}.
        # C_16(a, b, 8).
        # Try a=1, b=5?
        # Let's just use LUW for k=3 if specific constructions fail.
        return None

    # Portfolio Search
    candidates = []

    # Add specific known cages
    if k == 3 and g == 5:
        res = petersen_construct()
        if res: candidates.append(res)
    if k == 3 and g == 6:
        res = heawood_construct()
        if res: candidates.append(res)
    if k == 3 and g == 9:
        res = c39_construct()
        if res: candidates.append(res)
    
    # Try LUW for all cases
    res = luw_construct(k, g)
    if res:
        candidates.append(res)

    # Select best
    best_n, best_edges = None, None
    for n, edges in candidates:
        if best_n is None or n < best_n:
            best_n = n
            best_edges = edges

    if best_edges is None:
        # Fallback: Return a valid graph if possible, else error
        # LUW should always work for k>=3
        return luw_construct(k, g)
        
    return best_n, best_edges
