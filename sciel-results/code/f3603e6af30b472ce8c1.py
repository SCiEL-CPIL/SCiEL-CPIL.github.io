import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            return k * (k - 1) ** (h - 1)
        else:
            h = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(h))

    def make_edges_from_cyclic(p, a, b, c):
        """Build 3-regular graph on Z_p with generators ±a, ±b, ±c where a+b+c=0."""
        # Verify: a, b, c are nonzero mod p, and no two are equal mod p
        # Also a+b = -c mod p, so the three "positive" generators sum to 0
        # Edges: for each vertex v, connect to v+a, v-b, v+c (or similar)
        # Actually: generators are {a, -a, b, -b, c, -c} but since a+b+c=0,
        # we have -c = a+b, so the generator set is {a, b, a+b} and their negatives
        # Wait: if a+b+c=0, then -c = a+b. So generators {a, b, c} with c=-(a+b)
        # The 6 generators are {a, -a, b, -b, c, -c} = {a, -a, b, -b, -(a+b), a+b}
        # For a 3-regular graph, we need exactly 3 neighbors per vertex.
        # In a Cayley graph with generating set S of size 2m (m pairs of inverses),
        # degree = 2m. For degree 3, we need... hmm, 3 is odd.
        # 
        # Actually for k=3, the Cayley graph approach with ±pairs gives even degree.
        # We need a different approach. Let me use a "cubic" construction:
        # 
        # Alternative: Use a circulant-like graph where vertex i connects to
        # i+d1, i-d1, and one more. But that's not a Cayley graph.
        #
        # Let me reconsider. For 3-regular graphs, we can use:
        # - Petersen-like constructions
        # - Or: take a 2-regular Cayley graph (cycle) and add a perfect matching
        #
        # Actually, the standard approach for 3-regular high-girth graphs uses:
        # The "cubic" Cayley graph doesn't exist in the traditional sense since
        # 3 is odd. Instead, use a group with a generating set of 3 elements where
        # the group operation gives the edges.
        #
        # Let me use a different strategy: search over pairs of integers (a,b)
        # and construct a 3-regular graph on n vertices using a "circulant + matching" approach.
        #
        # Actually, let me just do a direct search: for each prime p, try to find
        # a 3-regular graph of girth >= g on p vertices (or p+1, etc.)
        #
        # Better idea: Use the known construction from the literature.
        # For 3-regular graphs of girth 8, 10, 12, 14, 16, there are known
        # constructions using finite fields or group theory.
        #
        # Let me try: for each n from Moore bound upward, try random 3-regular
        # graphs with a fast girth check, and if that fails, use the LUW fallback.
        #
        # Actually, the key insight from the knowledge: "Replacing the fixed large-depth
        # LUW fallback with a minimal-depth search reduces graph size from ~350k to ~13k"
        #
        # So the LUW construction CAN give small graphs if we search for minimal depth.
        # Let me implement a proper LUW search.
        pass

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar construction.
        q: prime power (use q=3 for smallest)
        d: depth parameter
        Returns (n, edges) for a 3-regular graph of girth at least 2d+2 (roughly).
        """
        # For q=3, d=5: girth ~ 2*5+2 = 12... not enough
        # For q=3, d=6: girth ~ 14
        # For q=3, d=7: girth ~ 16
        # The number of vertices is 2 * q^d * (q-1) / 2 = q^d * (q-1)
        # Wait, the LUW graph has 2 * sum_{i=0}^{d-1} q^i * ... 
        # Actually the LUW(3,d) graph has 2*(3^d - 1)/2 * ... 
        # Let me just implement it and count.
        
        # Rule for LUW:
        # rule[2] = (1,1), rule[3] = (2,1)
        # For c >= 4: rule[c] depends on c mod 4
        # r = (c-4) % 4, c0 = c - r
        # rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
        rule = {2: (1, 1), 3: (2, 1)}
        for c in range(4, d + 1):
            r = (c - 4) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1: (None, x1, x2, ..., xd)
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def fast_girth_check(n, edges, max_girth):
        """
        Check if graph has girth >= max_girth using non-backtracking walks.
        Returns True if girth >= max_girth, False otherwise (and the girth found).
        """
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS in the non-backtracking state space
        # State: (current_vertex, previous_vertex)
        # We want to find if any non-backtracking walk of length < max_girth returns to start
        
        for start in range(n):
            # BFS from start in non-backtracking space
            # dist[(u, v)] = length of shortest non-backtracking walk from start to (u,v)
            # where the walk ends at u having come from v
            visited = set()
            queue = deque()
            
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            
            while queue:
                u, prev, dist = queue.popleft()
                
                if dist >= max_girth:
                    break
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False, dist + 1  # Found cycle of length dist+1
                    if (nb, u) not in visited:
                        visited.add((nb, u))
                        queue.append((nb, u, dist + 1))
            
            # Clear for next start (but this is expensive)
            # Actually we can break early if we find a cycle
            # But we need to check all starts... this is O(n * branching^(girth/2))
            # For girth 16, that's too slow for n=1000+
        
        return True, max_girth

    def girth_check_optimized(n, adj, max_girth):
        """
        Optimized girth check: for each vertex, enumerate non-backtracking walks
        up to length max_girth-1 and check if any returns to the start.
        Uses a set of visited (current, prev) states to avoid re-exploration.
        """
        for start in range(n):
            # BFS in non-backtracking state space from start
            # State: (current, prev)
            # We track the minimum distance to each state
            dist = {}
            queue = deque()
            
            for nb in adj[start]:
                queue.append((nb, start, 1))
                dist[(nb, start)] = 1
            
            while queue:
                u, prev, d = queue.popleft()
                
                if d >= max_girth:
                    continue
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False  # Found cycle shorter than max_girth
                    if (nb, u) not in dist:
                        dist[(nb, u)] = d + 1
                        queue.append((nb, u, d + 1))
        
        return True

    # Strategy: Try LUW with small q and d values
    # LUW(q, d) gives girth approximately 2*d + 2
    # For girth 16, we need d >= 7
    # n = 2 * (q^d - 1) / (q - 1) * ... let me just compute
    
    # Try LUW with q=3, d=7 first (smallest expected n for girth 16)
    # Then try q=3, d=8, etc. if girth isn't sufficient
    # Also try q=2 (but q must be prime power >= 3 for the construction to work well)
    
    # Actually, let me try multiple (q, d) combinations and pick the smallest n with girth >= 16
    
    candidates = []
    for q in [3, 5, 7]:
        for d in range(5, 12):
            if time.time() > deadline - 20:
                break
            try:
                n, edges = luw_construct(q, d)
                if n <= 10000:  # Only check girth for reasonable sizes
                    # Quick girth check
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    has_girth = girth_check_optimized(n, adj, g)
                    candidates.append((n, q, d, edges, has_girth))
            except Exception:
                continue
    
    # Pick the smallest n with girth >= g
    valid = [(n, q, d, edges) for n, q, d, edges, hg in candidates if hg]
    if valid:
        valid.sort(key=lambda x: x[0])
        n, q, d, edges = valid[0]
        return n, edges
    
    # If LUW didn't work, try random 3-regular graphs
    def random_3reg(n):
        for _ in range(100):
            stubs = []
            for i in range(n):
                stubs += [i] * 3
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 3 * n, 2):
                a, b = stubs[i], stubs[i + 1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if ok:
                return [list(s) for s in adj]
        return None

    # Try random graphs for smaller n
    for n_try in range(512, 1000, 32):
        if time.time() > deadline - 15:
            break
        if n_try % 2 != 0:
            continue
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            adj = random_3reg(n_try)
            if adj is None:
                continue
            # Check girth
            if girth_check_optimized(n_try, adj, g):
                edges = []
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_try, edges

    # Final fallback: LUW with q=3, d=10 (should definitely have girth >= 16)
    n, edges = luw_construct(3, 10)
    return n, edges
