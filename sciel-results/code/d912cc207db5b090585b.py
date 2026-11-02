"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _compute_nibble_table(q):
    """For odd q, return nibble[x][y] = x^2 + x*y + y^2 mod q."""
    nibble = [[0] * q for _ in range(q)]
    for x in range(q):
        for y in range(q):
            nibble[x][y] = (x * x + x * y + y * y) % q
    return nibble


def _compute_nibble_table4():
    """For q=4, return nibble[x][y] = x^2 + x*y + y^2 in GF(4) via XOR."""
    # GF(4) = {0,1,2,3} with 1+1=0, etc. Multiplication:
    # 0*anything=0, 1*anything=anything, 2*2=3, 2*3=1, 3*3=2
    mul = [[0]*4 for _ in range(4)]
    mul[0] = [0,0,0,0]
    mul[1] = [0,1,2,3]
    mul[2] = [0,2,3,1]
    mul[3] = [0,3,1,2]
    def add(x, y): return x ^ y
    def square(x): return mul[x][x]
    def mul_xy(x, y): return mul[x][y]
    nibble = [[0]*4 for _ in range(4)]
    for x in range(4):
        for y in range(4):
            nibble[x][y] = add(square(x), add(mul_xy(x, y), square(y)))
    return nibble


def _verify_graph(n, edges, k, g, deadline):
    """Verify k-regularity and girth >= g. Returns True if valid."""
    if n <= 0:
        return False
    # Check no self-loops
    for u, v in edges:
        if u == v:
            return False
    # Check edge count
    if len(edges) != n * k // 2:
        return False
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Check girth using non-backtracking walk
    # A non-backtracking walk of length L from vertex v is a sequence v0=v, v1, ..., vL
    # where v_{i+1} != v_{i-1} for i >= 1.
    # The girth is the minimum length of a non-backtracking closed walk (vL = v0, L >= 1,
    # and v_{L-1} != v1 to avoid trivial 2-cycles).
    # Actually, for simple graphs, girth = min cycle length.
    # We check: for each vertex, do non-backtracking BFS and see if we return to start
    # at distance >= g.
    # More precisely: girth >= g iff no non-backtracking walk of length < g returns to start.
    # But we need to be careful: a non-backtracking walk of length L returning to start
    # corresponds to a cycle of length L (if L >= 3) or a 2-walk (if L=2, but that's not
    # allowed in simple graphs without multiple edges).
    
    # For each start vertex, track (current_vertex, previous_vertex) states.
    # If we reach (start, prev) where prev != start and step >= 3, we have a cycle.
    # Actually, we need: walk v0, v1, ..., vL with vL = v0, v_{i+1} != v_{i-1}, L >= 3.
    # This means at step L, we're at v0 with prev = v_{L-1}, and v_{L-1} != v1 (to ensure
    # non-backtracking at the end... wait, the non-backtracking condition is local).
    
    # Let's use: state = (current, prev). Start from (v, neighbor) for each neighbor.
    # Walk to (v, prev2) where prev2 != neighbor. If we reach (v, prev2), that's a cycle
    # of length (steps taken) where the walk went v -> neighbor -> ... -> prev2 -> v.
    # The cycle length is the number of edges in the walk.
    
    # Simpler: for each start vertex s, do BFS on states (cur, prev) starting from
    # (s, nb) for each neighbor nb of s. Track distance (number of edges traversed).
    # If we reach (s, prev) where prev != nb_start, we have a cycle.
    # Actually, the standard approach: 
    #   For each vertex s, for each neighbor nb of s:
    #     State: (cur, prev), start at (nb, s), distance 1.
    #     If we reach (s, prev) with prev != nb, the cycle length is distance+1? 
    #     Let me think again.
    
    # Walk: s -> nb -> ... -> cur -> s. Length = number of edges.
    # State after first step: (nb, s), edges used = 1.
    # If next step goes to s: (s, nb), edges used = 2. But this is the 2-walk, not a cycle.
    # We want: s -> nb -> ... -> x -> s where x != nb (to avoid backtracking nb->s->nb).
    # So we want to reach (s, x) where x != nb, with edges used >= 3.
    # The cycle length is edges used.
    
    # Let's do: for each s, for each nb in adj[s]:
    #   BFS from (nb, s) at dist=1.
    #   When we pop (cur, prev) at dist d, for each nxt in adj[cur] where nxt != prev:
    #     new_dist = d + 1
    #     If nxt == s and d+1 >= 3: found cycle of length d+1.
    #     Else, push (nxt, cur) at dist d+1 if not visited.
    
    # This is O(n * k^2 * g) in the worst case, which for n=1250, k=5, g=9 is fine.
    
    for s in range(n):
        # For each neighbor of s, do a non-backtracking BFS
        for nb in adj[s]:
            # State: (cur, prev), dist = number of edges from s to cur in this walk
            # Start: (nb, s) at dist 1
            visited = set()
            queue = [(nb, s, 1)]
            visited.add((nb, s))
            while queue:
                if time.time() > deadline:
                    return False  # timeout, assume invalid
                cur, prev, dist = queue.pop(0)
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    new_dist = dist + 1
                    if nxt == s:
                        if new_dist >= 3:
                            return False  # cycle of length new_dist < g would be caught below
                        # new_dist == 2 means we went s->nb->s, not a real cycle
                    else:
                        if new_dist < g:
                            state = (nxt, cur)
                            if state not in visited:
                                visited.add(state)
                                queue.append((nxt, cur, new_dist))
                        # if new_dist >= g, no need to expand (we only care about cycles < g)
    return True


def construct(k, g):
    deadline = time.time() + 120.0
    q = k
    
    # Compute nibble table
    if q == 4:
        nibble = _compute_nibble_table4()
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
    else:
        nibble = _compute_nibble_table(q)
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
    
    # Try different d values, starting from smallest
    # For LUW CD(d, q), the girth is at least 2d+2 (even, since bipartite)
    # We need girth >= g, so 2d+2 >= g, i.e., d >= (g-2)/2
    # But the lower bound d+5 from the paper is looser.
    # Let's just try d values and verify.
    
    min_d = max(2, (g - 2 + 1) // 2)  # ceil((g-2)/2)
    
    best_result = None
    
    for d in range(min_d, min_d + 20):
        if time.time() > deadline - 30:
            break
        
        # Build CD(d, q)
        # Vertices: (side, x0, x1, ..., xd) where side in {0, 1}
        # side=0: points, side=1: lines
        # Point p = (0, p1, p2, ..., pd) where p1 is free (0..q-1), p2..pd computed
        # Line l = (1, l1, l2, ..., ld)
        # Actually, the standard LUW construction:
        # Points: (a, b, c1, c2, ..., c_{d-1}) where a,b in GF(q), c_i in GF(q)
        # Lines: (a', b', l1, l2, ..., l_{d-1})
        # Hmm, this is getting complicated. Let me use the original code's approach.
        
        # The original code uses:
        # Vertex: (side, (None, x1, x2, ..., xd))
        # For side=0 (points): x1 is free (0..q-1), x2..xd computed
        # For side=1 (lines): x1 is free, x2..xd computed
        # Edge between point p and line l iff for all c in 2..d:
        #   l[c] - p[c] = l[a_c] * p[b_c] where (a_c, b_c) = rule[c]
        
        # The rule depends on d. Let me reconstruct it.
        # For the standard LUW, the rule is:
        # For c=2: a=1, b=1 (i.e., l[2] - p[2] = l[1] * p[1])
        # For c=3: a=2, b=1 (l[3] - p[3] = l[2] * p[1])
        # For c=4: a=1, b=2 (l[4] - p[4] = l[1] * p[2])
        # For c>=5: pattern repeats with period 4
        #   c mod 4 == 1: (1, c-2)
        #   c mod 4 == 2: (c-1, 1)
        #   c mod 4 == 3: (c, 1)  -- wait, this doesn't look right
        #   c mod 4 == 0: (1, c+1) -- also doesn't make sense
        
        # Let me just use the original rule from the seed code.
        rule = {}
        if d >= 2:
            rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            elif r == 3:
                rule[c] = (1, c0 + 1)
        
        # Build graph via BFS from zero point
        # Point: (0, (None, x1, x2, ..., xd))
        # Line: (1, (None, x1, x2, ..., xd))
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], y[a] * x[b] % q if q != 4 else nibble[y[a]][x[b]])
                    else:
                        y[c] = sub(x[c], y[a] * x[b] % q if q != 4 else nibble[y[a]][x[b]])
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        
        # Verify
        if _verify_graph(n, edges, k, g, deadline):
            return n, edges
        else:
            best_result = (n, edges)
    
    # If verification failed for all d, try a random k-regular graph
    # This is a fallback
    n_try = max(2 * k * g, 50)
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        # Generate random k-regular graph
        # Use configuration model
        while True:
            if time.time() > deadline - 5:
                break
            # Create stubs
            stubs = list(range(n_try)) * k
            random.shuffle(stubs)
            # Pair stubs
            edges_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges_set:
                    valid = False
                    break
                edges_set.add((min(u,v), max(u,v)))
            if valid and len(edges_set) == n_try * k // 2:
                edges = list(edges_set)
                if _verify_graph(n_try, edges, k, g, deadline):
                    return n_try, edges
    
    if best_result:
        return best_result
    
    # Absolute fallback: return a small invalid graph (should not happen)
    return 2 * k, [(i, i + 1) for i in range(k)]
# EVOLVE-BLOCK-END
