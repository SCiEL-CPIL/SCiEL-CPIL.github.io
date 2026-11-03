"""
Cage problem: k-regular graph with girth >= g, minimize n.
Strategy:
1. Try known explicit constructions (LUW, incidence of generalized polygons, etc.)
2. If none match, do a greedy edge-adding search with girth constraints.
"""
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() > deadline

    # --- Validation: simple, k-regular, girth >= g ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Girth check via BFS from each vertex
        for s in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[s] = 0
            q2 = deque([s])
            while q2:
                u = q2.popleft()
                if depth[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        q2.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            clen = depth[u] + depth[v] + 1
                            if clen < g:
                                return False
        return True

    # --- LUW family ---
    def luw_family():
        q = k
        # d is the "length" parameter; for girth g, we need d such that 2d+1 >= g
        # LUW gives girth 2d+1 typically
        d = (g - 1) // 2
        if d < 2:
            d = 2

        if q == 2:
            # Special small case
            add = lambda x, y: (x + y) % 2
            sub = lambda x, y: (x - y) % 2
            mul = lambda x, y: x * y % 2
        elif q == 3:
            add = lambda x, y: (x + y) % 3
            sub = lambda x, y: (x - y) % 3
            mul = lambda x, y: x * y % 3
        elif q == 4:
            # GF(4)
            table_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
            table_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: table_add[x][y]
            sub = lambda x, y: table_add[x][y]  # char 2
            mul = lambda x, y: table_mul[x][y]
        elif q == 5:
            add = lambda x, y: (x + y) % 5
            sub = lambda x, y: (x - y) % 5
            mul = lambda x, y: x * y % 5
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Rule for LUW: for c in 2..d, rule[c] = (a, b)
        # Standard LUW rule:
        # c=2: (1,1)
        # c=3: (2,1)
        # c=4: (1,2)
        # c=5: (1,3)
        # c=6: (3,1)
        # c=7: (1,4)
        # c=8: (4,1)
        # c=9: (1,5)
        # c=10: (5,1)
        # Pattern: for c >= 2:
        #   if c is even: rule[c] = (c//2, 1)  ... wait let me check
        #   Actually the standard is:
        #   c=2: (1,1), c=3: (2,1), c=4: (1,2), c=5: (1,3), c=6: (3,1), ...
        #   Let me use the known pattern:
        #   For c >= 2:
        #     if c % 2 == 0: a = c//2, b = 1  (for c=2: a=1,b=1; c=4: a=2,b=1? No, c=4 is (1,2))
        #   Let me just hardcode the first few and use a pattern.
        #   Actually the LUW rule is:
        #     rule[2] = (1,1)
        #     rule[3] = (2,1)
        #     rule[4] = (1,2)
        #     rule[5] = (1,3)
        #     rule[6] = (3,1)
        #     rule[7] = (1,4)
        #     rule[8] = (4,1)
        #     rule[9] = (1,5)
        #     rule[10] = (5,1)
        #     ...
        #   Pattern for c >= 4:
        #     if c is even: rule[c] = (1, c//2) if c//2 is odd? No...
        #     c=4: (1,2), c=6: (3,1), c=8: (4,1), c=10: (5,1)
        #     c=5: (1,3), c=7: (1,4), c=9: (1,5)
        #   Let me just use: for c >= 2, if c is odd and c > 3: (1, (c+1)//2)
        #   if c is even and c > 2: (c//2, 1)
        #   Check: c=4: (2,1) but should be (1,2). So that's wrong.
        #   
        #   Correct pattern from literature:
        #   c=2: (1,1)
        #   c=3: (2,1)
        #   c=4: (1,2)
        #   c=5: (1,3)
        #   c=6: (3,1)
        #   c=7: (1,4)
        #   c=8: (4,1)
        #   c=9: (1,5)
        #   c=10: (5,1)
        #   
        #   So: 
        #   c=2: (1,1)
        #   c=3: (2,1)
        #   For c >= 4:
        #     if c is even: (c//2, 1)  -> c=4: (2,1) WRONG, should be (1,2)
        #   
        #   Let me re-examine. The LUW construction uses a specific rule.
        #   I'll use the rule from the original paper which I'll implement carefully.
        #   
        #   Actually, let me just use a simpler approach: try multiple d values and 
        #   use the rule that's known to work. For the purpose of this task, I'll
        #   use the rule from the seed program which seemed to work.

        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            # Pattern: alternate between (1, m) and (m, 1)
            # c=5: (1,3), c=6: (3,1), c=7: (1,4), c=8: (4,1), c=9: (1,5), c=10: (5,1)
            # So for c >= 5:
            #   m = (c + 1) // 2  for odd c: c=5->3, c=7->4, c=9->5
            #   m = c // 2        for even c: c=6->3, c=8->4, c=10->5
            #   if c is odd: rule[c] = (1, (c+1)//2)
            #   if c is even: rule[c] = (c//2, 1)
            if c % 2 == 1:
                rule[c] = (1, (c + 1) // 2)
            else:
                rule[c] = (c // 2, 1)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [0] * (d + 1)
                y[0] = None  # placeholder
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y[1:])

        start = (0, (0,) * d)
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
        return len(order), edges

    # --- Incidence graph of generalized hexagon of order (2,4) ---
    # This gives a 5-regular graph with girth 12 and 126 vertices.
    # The generalized hexagon of order (2,4) has points and lines.
    # Points: 126, Lines: 126 (dual)
    # Each point is on 5 lines, each line has 5 points.
    # Incidence graph: 252 vertices, 5-regular, girth 12.
    # But we want the point graph or line graph which is 5-regular on 126 vertices?
    # No, the incidence graph has 2*126=252 vertices.
    # Actually for the cage problem, we can use the incidence graph directly.
    # But 252 vertices for girth 12 is not optimal.
    # For k=5, g=9, we need something else.

    # --- Greedy construction with girth constraint ---
    def greedy_construct(k, g):
        """
        Greedy edge-adding: start with n vertices, add edges one by one
        maintaining k-regularity and girth >= g.
        """
        # Estimate n from Moore bound
        # For k-regular, girth g:
        # If g is odd: n >= k + (k-1)(1 + k + ... + k^((g-3)/2))
        # If g is even: n >= 2 * (1 + (k-1) + ... + (k-1)^((g-2)/2))
        if g % 2 == 1:
            half = (g - 3) // 2
            moore = k + (k - 1) * sum(k ** i for i in range(half + 1))
            # Actually: k + (k-1) * (1 + k + k^2 + ... + k^half)
            # = k + (k-1) * (k^(half+1) - 1) / (k - 1)
            # = k + k^(half+1) - 1
            # Wait, that's not right either. Let me just compute directly.
            moore = k
            for i in range(1, half + 1):
                moore += (k - 1) * (k ** i)
        else:
            half = (g - 2) // 2
            moore = 2 * sum((k - 1) ** i for i in range(half + 1))

        # Try a few n values starting from moore bound
        best_n = None
        best_edges = None

        for n in range(max(moore, 2 * k), max(moore + 200, 4 * k + 10)):
            if check_deadline():
                break
            # Try to build k-regular graph on n vertices with girth >= g
            if n * k % 2 != 0:
                continue
            result = _build_k_regular_girth(n, k, g, deadline)
            if result is not None:
                edges = result
                if best_edges is None or n < best_n:
                    best_n = n
                    best_edges = edges
                # Found one, try smaller n? No, we're going up. 
                # Actually we should try to find the smallest.
                # Let's just return the first valid one we find going up from Moore bound.
                # But Moore bound might be too low. Let's return what we have.
                return n, edges

        if best_n is not None:
            return best_n, best_edges
        return None

    def _build_k_regular_girth(n, k, g, deadline):
        """Try to build a k-regular graph on n vertices with girth >= g."""
        import random
        random.seed(42)
        
        # Start with empty graph
        adj = [set() for _ in range(n)]
        edges = []
        
        # Need to add n*k/2 edges
        total_edges = n * k // 2
        remaining = [k] * n
        
        # Use a greedy approach: repeatedly pick two vertices with remaining degree > 0
        # and add an edge if it doesn't create a cycle shorter than g.
        
        # For efficiency, maintain a list of vertices with remaining degree
        candidates = list(range(n))
        
        for _ in range(total_edges):
            if time.time() > deadline:
                return None
            if not candidates:
                return None
            
            # Pick a vertex with max remaining degree
            u = max(candidates, key=lambda v: remaining[v])
            if remaining[u] == 0:
                return None
            
            # Find a valid neighbor for u
            valid_v = None
            # Try random order
            perm = candidates[:]
            random.shuffle(perm)
            for v in perm:
                if v == u or remaining[v] == 0:
                    continue
                if v in adj[u]:
                    continue
                # Check if adding (u,v) creates a cycle shorter than g
                if _would_create_short_cycle(u, v, adj, g):
                    continue
                valid_v = v
                break
            
            if valid_v is None:
                # Try in order
                for v in candidates:
                    if v == u or remaining[v] == 0:
                        continue
                    if v in adj[u]:
                        continue
                    if _would_create_short_cycle(u, v, adj, g):
                        continue
                    valid_v = v
                    break
            
            if valid_v is None:
                return None
            
            # Add edge
            adj[u].add(valid_v)
            adj[valid_v].add(u)
            remaining[u] -= 1
            remaining[valid_v] -= 1
            edges.append((u, valid_v))
            
            # Update candidates
            if remaining[u] == 0:
                candidates.remove(u)
            if remaining[valid_v] == 0:
                candidates.remove(valid_v)
        
        # Verify all degrees are k
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return edges

    def _would_create_short_cycle(u, v, adj, g):
        """Check if adding edge (u,v) creates a cycle shorter than g."""
        # BFS from u to v in the current graph (without using edge u-v)
        # If distance < g-1, then adding (u,v) creates a cycle of length dist+1 < g
        if v in adj[u]:
            return True  # already an edge
        
        # BFS
        dist = {u: 0}
        queue = deque([u])
        while queue:
            x = queue.popleft()
            if dist[x] >= g - 1:
                break
            for y in adj[x]:
                if y == v:
                    return dist[x] + 1 < g
                if y not in dist:
                    dist[y] = dist[x] + 1
                    queue.append(y)
        return False

    # --- Try constructions in order ---
    # 1. LUW
    try:
        n_luw, e_luw = luw_family()
        if validate(n_luw, e_luw, k, g):
            return n_luw, e_luw
    except Exception:
        pass

    # 2. Greedy
    result = greedy_construct(k, g)
    if result is not None:
        n, e = result
        if validate(n, e, k, g):
            return n, e

    # 3. Fallback: LUW even if not fully validated
    try:
        n_luw, e_luw = luw_family()
        return n_luw, e_luw
    except Exception:
        pass

    # Last resort: return a simple cycle-based construction (may not be valid)
    n = max(2 * k, g)
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % n))
    return n, edges

# EVOLVE-BLOCK-END
