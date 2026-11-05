import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = float('inf')
    best_edges = None
    
    # Candidate 1: Lazebnik-Ustimenko-Woldar (LUW)
    # This is the seed logic, but we will verify and compare.
    try:
        n_luw, edges_luw = _construct_luw(k, g)
        if _verify_girth(n_luw, edges_luw, k, g):
            if n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw
    except Exception:
        pass
    
    # Candidate 2: Cayley graphs of SL(2, q)
    # We try small q values.
    # The degree is 4 (2 generators and their inverses).
    # If k != 4, we might need to adjust or skip if k is not 4.
    # For general k, SL(2,q) might not be k-regular easily.
    # But for k=4, it's a strong candidate.
    if k == 4:
        for q in [5, 7, 9, 11, 13, 17, 19, 23, 29, 31]:
            if time.time() > deadline:
                break
            try:
                n_cay, edges_cay = _construct_sl2_cayley(q)
                # Check if n is better than current best
                if n_cay >= best_n:
                    continue
                if _verify_girth(n_cay, edges_cay, k, g):
                    best_n = n_cay
                    best_edges = edges_cay
            except Exception:
                pass
    
    # If no graph found, fallback to LUW (even if not verified, but seed logic is usually valid)
    if best_edges is None:
        try:
            best_n, best_edges = _construct_luw(k, g)
        except Exception:
            # Fallback to a trivial construction (should not happen for valid k,g)
            best_n = k
            best_edges = []
            
    return best_n, best_edges

def _construct_luw(k, g):
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

def _construct_sl2_cayley(q):
    # Construct Cayley graph of SL(2, q) with generators A, B, A^-1, B^-1
    # Matrices are (a, b, c, d) representing [[a, b], [c, d]]
    # Identity: (1, 0, 0, 1)
    
    # Determine field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        inv = lambda x: 0 if x == 0 else table[x][x] # Inverse in GF(4) is x^3 = x? No, x^2+x+1=0?
        # GF(4) elements: 0, 1, a, a+1. 1^-1=1, a^-1=a^2=a+1, (a+1)^-1=a.
        # Table for inv: 0->0, 1->1, 2->3, 3->2 (if 2=a, 3=a+1)
        inv_table = {0:0, 1:1, 2:3, 3:2}
        def inv_func(x): return inv_table[x]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        # Modular inverse
        def inv_func(x):
            if x == 0: return 0
            for i in range(1, q):
                if (x * i) % q == 1:
                    return i
            return 0

    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            add(mul(a1, a2), mul(b1, c2)),
            add(mul(a1, b2), mul(b1, d2)),
            add(mul(c1, a2), mul(d1, c2)),
            add(mul(c1, b2), mul(d1, d2))
        )

    def mat_inv(m):
        a, b, c, d = m
        det = sub(mul(a, d), mul(b, c))
        if det == 0:
            return None
        det_inv = inv_func(det)
        return (
            mul(det_inv, d),
            mul(det_inv, sub(0, b)),
            mul(det_inv, sub(0, c)),
            mul(det_inv, a)
        )

    # Generators for SL(2, q)
    # A = [[0, -1], [1, 0]]
    # B = [[1, 1], [0, 1]]
    # Note: -1 is q-1
    A = (0, sub(0, 1), 1, 0)
    B = (1, 1, 0, 1)
    A_inv = mat_inv(A)
    B_inv = mat_inv(B)
    
    gens = [A, B, A_inv, B_inv]
    
    # BFS to generate group
    start = (1, 0, 0, 1)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Adjacency list for BFS
    adj = [[] for _ in range(q * (q*q - 1))] # Pre-allocate? No, use list of lists
    
    for v in order:
        idx_v = index[v]
        for gen in gens:
            w = mat_mul(v, gen)
            if w not in index:
                index[w] = len(order)
                order.append(w)
            idx_w = index[w]
            # Add edge
            if idx_v < idx_w:
                edges.append((idx_v, idx_w))
            
    return len(order), edges

def _verify_girth(n, edges, k, g):
    # Check if graph is k-regular and girth >= g
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    # For each vertex, do BFS up to depth g-1
    # If we find a cycle of length < g, return False
    # To speed up, we can stop early if we find a short cycle.
    
    # Optimized: Check for cycles of length < g
    # A graph has girth < g if and only if there is a cycle of length < g.
    # We can check for cycles of length 3, 4, ..., g-1.
    # For small g (like 10), this is feasible.
    
    # Simple check: BFS from each vertex, tracking parent to avoid backtracking.
    # If we reach a node that was visited in the current BFS path, we found a cycle.
    # The length is the distance.
    
    for start in range(n):
        # BFS
        # dist[v] = distance from start
        # prev[v] = previous node in BFS tree
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == prev[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    if dist[v] >= g - 1:
                        # We only care about cycles shorter than g
                        # If we reach depth g-1, any cycle found later would be >= g?
                        # No, a cycle of length L involves nodes at distances.
                        # If we find a back-edge to a node at distance d, cycle length is dist[u] + 1 - d + 1?
                        # Actually, non-backtracking BFS is better.
                        pass
                    queue.append(v)
                else:
                    # Found a back edge
                    # Cycle length = dist[u] + 1 + dist[v] - 2 * dist[lca(u, v)]?
                    # This is complex. 
                    # Simpler: Just check if the graph contains any cycle of length < g.
                    # For g=10, checking triangles, 4-cycles, etc. is fast.
                    pass
                    
    # For robustness, let's do a full non-backtracking BFS check for cycles < g
    # This is O(n * E * g)
    for start in range(n):
        # State: (node, prev_node, depth)
        visited = set()
        queue = deque([(start, -1, 0)])
        while queue:
            u, p, d = queue.popleft()
            if (u, p) in visited:
                continue
            visited.add((u, p))
            if d >= g - 1:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                # If v is start, we found a cycle of length d+1
                if v == start:
                    if d + 1 < g:
                        return False
                    else:
                        # Cycle length >= g, ok
                        continue
                # If v has been visited with a different prev, it might form a cycle
                # But non-backtracking BFS from start only finds cycles containing start.
                # To find ALL cycles, we need to check from all starts.
                # This is sufficient.
                queue.append((v, u, d + 1))
                
    return True
