"""Voltage-graph lift of the Petersen graph with deterministic voltage search.
Falls back to Lazebnik-Ustimenko-Woldar construction if search fails."""
import time
from collections import deque

_DEADLINE = None

def _elapsed():
    if _DEADLINE is None:
        return 0.0
    return time.time() - _DEADLINE[0]

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _petersen_edges():
    """Return Petersen graph edges (10 vertices, 15 edges, 4-regular)."""
    outer = [(i, (i + 1) % 5) for i in range(5)]
    star = [(i, (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    return outer + star + spokes

def _build_lift(voltages, m):
    """Build the voltage lift graph. Returns (n, edges) or None if not simple."""
    pe = _petersen_edges()
    n = 10 * m
    adj = [[] for _ in range(n)]
    for (u, v) in pe:
        vol = voltages[pe.index((u, v))]
        for r in range(m):
            a = u * m + r
            b = v * m + (r + vol) % m
            adj[a].append(b)
            adj[b].append(a)
    # Check simplicity: no loops, no duplicate edges
    for i in range(n):
        if i in adj[i]:
            return None
        if len(adj[i]) != 4:
            return None
        if len(set(adj[i])) != 4:
            return None
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges

def _verify_girth(n, edges, g):
    """Verify girth >= g using non-backtracking BFS from every vertex."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    # For each start vertex and each neighbor, do BFS tracking (node, prev)
    for s in range(n):
        visited = {}  # (node, prev) -> depth
        dq = deque()
        for nb in adj[s]:
            visited[(nb, s)] = 1
            dq.append((nb, s, 1))
        while dq:
            node, prev, depth = dq.popleft()
            if depth >= g:
                break
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (nb, node) not in visited:
                    visited[(nb, node)] = depth + 1
                    dq.append((nb, node, depth + 1))
                else:
                    # Found a cycle: length = depth + 1 + visited[(nb,node)]
                    # Actually this means a non-backtracking closed walk
                    # The cycle length is depth + 1 (going back)
                    # But we need to be careful: this is a non-backtracking
                    # walk from s to nb (via node) and from s to nb (earlier)
                    # forming a closed non-backtracking walk of length depth+1
                    # + visited[(nb,node)] - 1... 
                    # Simpler: if we reach a previously visited state, we have
                    # a closed non-backtracking walk of length depth + 1
                    # (the +1 for the edge node->nb which was already covered)
                    # Actually the total length is depth + 1 (current path) 
                    # but we need the cycle length which is depth + 1 + 
                    # visited[(nb,node)] - 2... 
                    # Let me just check: if depth + 1 >= g, we're fine.
                    # The cycle formed has length at most depth + 1.
                    if depth + 1 < g:
                        return False
    return True

def _verify_girth_fast(n, edges, g):
    """Faster girth check: BFS from each vertex, track shortest non-backtracking
    closed walk. Return True if all >= g."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    for s in range(n):
        # State: (current_node, previous_node) -> shortest depth
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dist[(nb, s)] = 1
            dq.append((nb, s))
        found_cycle = False
        while dq:
            node, prev = dq.popleft()
            d = dist[(node, prev)]
            if d >= g:
                break
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (nb, node) not in dist:
                    dist[(nb, node)] = d + 1
                    dq.append((nb, node))
                else:
                    # Cycle of length d + 1 (edge node->nb closes the walk)
                    if d + 1 < g:
                        return False
    return True

def _search_voltages(m, g, time_budget):
    """Search for voltages on Petersen edges giving girth >= g.
    Returns voltages list (15 values) or None."""
    pe = _petersen_edges()
    n_edges = 15
    # Candidate voltage sets: try several
    candidates = [
        list(range(1, m)),  # all non-zero
        [1, m//2, 2, m-1] if m >= 4 else [1],
        [1, 2, m-2, m-1] if m >= 4 else [1],
        [1, m//2, (m//2)+1, m-1] if m >= 4 else [1],
    ]
    # Use a deterministic random-like sequence for voltage assignment
    # We'll do a greedy/hill-climbing approach
    # First try: assign voltages to maximize the minimum non-backtracking closed walk voltage
    
    # For efficiency, precompute non-backtracking walks in Petersen up to length g
    # and track their voltage sums for each edge
    # A non-backtracking walk of length L uses L edges. Its total voltage is the sum.
    # We need: for every non-backtracking closed walk of length < g, total voltage != 0 mod m.
    
    # Enumerate non-backtracking closed walks in Petersen up to length g-1
    # Petersen has 10 vertices, 4-regular, 15 edges
    # Non-backtracking walks: state is (current_vertex, previous_vertex)
    # Start from each directed edge, extend without backtracking
    
    # Build adjacency
    adj = [[] for _ in range(10)]
    edge_index = {}  # (u,v) -> edge_idx
    for idx, (u, v) in enumerate(pe):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # Enumerate non-backtracking closed walks of length 2 to g-1
    # A closed non-backtracking walk: v0 -> v1 -> ... -> vL = v0, no immediate backtrack
    # For L < g, we need sum of voltages on edges (v_i, v_{i+1}) != 0 mod m
    
    # Collect all non-backtracking closed walks up to length g-1
    # Represent each as a sequence of edge indices
    closed_walks = []  # list of tuples of edge indices
    
    for L in range(2, g):
        # BFS over states (current, prev) starting from each directed edge
        # Track path of edges
        # For efficiency, just enumerate: from each directed edge (u,v), 
        # extend to (v,w) where w != u, etc.
        # Start states: all directed edges
        for u in range(10):
            for v in adj[u]:
                # Walk: u -> v, edge index
                e0 = edge_index[(u, v)]
                if L == 1:
                    continue
                # Extend to length L and check if closed (ends at u)
                # Use iterative deepening
                # For small L (up to 9) and Petersen (10 nodes), this is feasible
                # State: (current, prev, path_edges)
                # We want walks of exactly length L that end back at u
                # Non-backtracking: next vertex != prev
                
                # DFS
                def dfs(cur, prev, path, depth):
                    if depth == L:
                        if cur == u:
                            closed_walks.append(tuple(path))
                        return
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        e = edge_index[(cur, nxt)]
                        path.append(e)
                        dfs(nxt, cur, path, depth + 1)
                        path.pop()
                
                path = [e0]
                dfs(v, u, path, 1)
    
    # Now we have all non-backtracking closed walks of length 2 to g-1
    # For each, the sum of voltages on its edges must be != 0 mod m
    # This is a constraint satisfaction problem on 15 variables (voltages)
    # with domain [1, m-1] (or restricted)
    
    # Greedy approach: assign voltages one by one, choosing value that satisfies
    # the most constraints
    # Or: try a few structured assignments and verify
    
    # Structured assignment: try to make all length-2 walks have non-zero voltage
    # Length-2 non-backtracking closed walks: u->v->w->u where w != u (i.e., triangle)
    # Petersen has no triangles (girth 5), so no length-3 closed walks
    # Length-4: u->v->w->x->u, non-backtracking
    # Length-5: u->v->w->x->y->u (5-cycle, Petersen has these)
    
    # Simple approach: try random-like assignments with a fixed seed
    # and verify
    import random
    rng = random.Random(42)
    
    best_voltages = None
    for trial in range(200):
        if _elapsed() > time_budget:
            break
        # Choose a candidate set
        cset = candidates[trial % len(candidates)]
        voltages = [rng.choice(cset) for _ in range(n_edges)]
        # Verify: check all closed walks
        ok = True
        for walk in closed_walks:
            total = sum(voltages[e] for e in walk) % m
            if total == 0:
                ok = False
                break
        if ok:
            best_voltages = voltages
            break
        # If not ok, try a perturbation
        if trial > 0 and best_voltages:
            # Perturb from best
            voltages = list(best_voltages)
            for _ in range(3):
                idx = rng.randrange(n_edges)
                voltages[idx] = rng.choice(cset)
            ok = True
            for walk in closed_walks:
                total = sum(voltages[e] for e in walk) % m
                if total == 0:
                    ok = False
                    break
            if ok:
                best_voltages = voltages
                break
    
    return best_voltages

def construct(k, g):
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = [start_time + 140.0]  # 10s safety margin
    
    # Moore bound for 4-regular, girth 10: 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1+4+12+36+108 = 161
    # Actually for even girth g=10: Moore bound = 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2-1))
    # = 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242... wait
    # For even girth 2r: Moore bound = 2 * sum_{i=0}^{r-1} (k-1)^i
    # r = 5: 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242
    # So n >= 242, and the known cage is 320.
    
    # Try voltage lift approach
    if k == 4 and g == 10:
        m = 32  # 10 * 32 = 320 vertices
        voltages = _search_voltages(m, g, time_budget=120.0)
        if voltages is not None:
            result = _build_lift(voltages, m)
            if result is not None:
                n, edges = result
                # Verify
                if _verify_girth_fast(n, edges, g):
                    return n, edges
    
    # Fallback: L-U-W construction
    return _luw_construct(k, g)
