"""Construct a 5-regular graph of girth 9 using algebraic incidence over GF(5)."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 110.0
    q = 5
    # Bipartite graph: points = GF(q)^3, lines = "planes" parameterized by (a,b,c)
    # Point (x,y,z) ~ Line (a,b,c) iff x + a*y + b*z + c = 0  (over GF(q))
    # Each point lies on q lines, each line contains q points -> q-regular.
    # Girth analysis: cycles in this incidence graph have length 2*len.
    # A 4-cycle would need two points on two lines with both pairs satisfying the
    # linear equations, which forces the lines to be identical (no 4-cycles).
    # 6-cycles: need 3 points, 3 lines forming a triangle in incidence.
    # For q=5, girth of this incidence graph is 6 (there are 6-cycles).
    # So this alone gives girth 6, not 9. Need a lift.
    #
    # Alternative: use a 2-step construction. Build a base graph with girth >=9
    # directly using a larger field or a different incidence.
    #
    # Let's try: points = GF(q)^3, lines = triples (u, v, w) with the relation
    # point (x,y,z) ~ line (u,v,w) iff  u*x + v*y + w*z = 1  AND  x+y+z = u+v+w
    # This is more restrictive. But counting: each line has ~q^2 points? No.
    #
    # Simpler robust approach: use the known "M" construction for (5,9) cage candidates.
    # We build a bipartite 5-regular graph via a voltage lift of a small base.
    #
    # Base: a 5-regular multigraph-free structure. Use a circulant-like base of 2*m
    # vertices and lift by Z_m.
    #
    # Concretely: define base vertices B = {0,1,2,3,4,5} (6 base vertices, 3 per side).
    # We will lift by group G = Z_m (cyclic). Total vertices = 6*m.
    # For k=5, each base vertex has 5 "lifted" neighbors via voltage assignments.
    #
    # Let base side A = {a0,a1,a2}, side B = {b0,b1,b2}.
    # Each a_i connects to 5 base-b vertices (with multiplicities allowed in base,
    # resolved by voltages in the lift). We assign voltages in Z_m.
    #
    # To get girth >= 9, we need no closed non-backtracking walk of length < 9
    # in the lift. We search over voltage assignments.
    #
    # For efficiency, we try a parametric family and verify girth via BFS.
    
    # Parametric voltage assignment:
    # a0 -> b0:0, b1:1, b2:2, b0:3, b1:4   (5 edges from a0)
    # a1 -> b1:0, b2:1, b0:2, b1:3, b2:4
    # a2 -> b2:0, b0:1, b1:2, b2:3, b0:4
    # This gives a 5-regular bipartite base (with multi-edges).
    # Lift by Z_m: vertex (a_i, t) ~ (b_j, t + voltage) for each base edge.
    
    def build_lift(m, voltages):
        """Build the lifted graph. voltages[i][j] = list of (base_neighbor, voltage)."""
        n_base = 6  # 3 A + 3 B
        n = n_base * m
        edges = []
        seen = set()
        for i in range(3):
            for t in range(m):
                a_idx = i
                for (j, v) in voltages[i]:
                    b_idx = 3 + j
                    t2 = (t + v) % m
                    u = a_idx * m + t
                    w = b_idx * m + t2
                    if u < w:
                        key = (u, w)
                    else:
                        key = (w, u)
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
        return n, edges
    
    def check_girth(n, edges, girth):
        """Check girth >= girth via BFS for shortest cycle. Returns True if girth >= girth."""
        from collections import deque
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # For each vertex, BFS tracking parent to detect cycles
        # Use non-backtracking BFS up to depth (girth-1)//2
        max_depth = (girth - 1) // 2
        for start in range(n):
            # BFS from start, tracking (node, parent) to avoid backtracking
            visited = {}  # (node, parent) -> depth
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= girth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == start:
                        # Found a cycle of length depth+1
                        if depth + 1 < girth:
                            return False
                    else:
                        if (nb, node) not in visited:
                            visited[(nb, node)] = depth + 1
                            if depth + 1 < max_depth:
                                queue.append((nb, node, depth + 1))
        return True
    
    # Try increasing m until girth >= 9
    # The voltage family above
    base_voltages = [
        [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4)],
        [(1, 0), (2, 1), (0, 2), (1, 3), (2, 4)],
        [(2, 0), (0, 1), (1, 2), (2, 3), (0, 4)],
    ]
    
    best = None
    for m in range(50, 300):
        if time.time() > deadline - 20:
            break
        n, edges = build_lift(m, base_voltages)
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            continue
        # Check girth
        if check_girth(n, edges, g):
            best = (n, edges)
            break
    
    if best is not None:
        return best
    
    # Fallback: LUW construction (original)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y),(lambda x,y: x^y),(lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y: (x+y)%q),(lambda x,y: (x-y)%q),(lambda x,y: x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0]==0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
