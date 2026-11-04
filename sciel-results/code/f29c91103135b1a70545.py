import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    if k != 4:
        return 0, []

    # Try voltage graph construction with K_{4,4} base (8 vertices)
    result = _voltage_graph_construct(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify(n, edges, k, g):
            return n, edges

    # Fallback: Lazebnik-Ustimenko-Woldar construction
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        return 0, []
        
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
                
    n = len(order)
    edges_list = sorted(set(edges))
    return n, edges_list


def _voltage_graph_construct(k, g, deadline):
    """Construct a 4-regular girth-g graph using voltage graph (lift) over Z_m."""
    if k != 4:
        return None

    best_result = None
    best_n = float('inf')

    # Base graph: K_{4,4} (8 vertices, 4-regular, girth 4)
    base_edges = _get_k44_edges()
    base_n = 8

    # Try different m values
    for m in range(2, 200):
        if time.time() > deadline - 10:
            break
        n = m * base_n
        if n >= best_n:
            continue
            
        # Try to find voltage assignment with girth >= g
        result = _optimize_voltages(base_edges, m, g, deadline - 5)
        if result is not None:
            voltages = result
            lift_edges = _build_lift(base_edges, voltages, m)
            if _verify(n, lift_edges, k, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, lift_edges)

    return best_result


def _get_k44_edges():
    """Get edges of K_{4,4}: vertices 0-3 on one side, 4-7 on other."""
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, j + 4))
    return edges


def _optimize_voltages(base_edges, m, g, deadline):
    """Find voltage assignment so lift has girth >= g using simulated annealing."""
    n_edges = len(base_edges)
    base_n = len(set(v for e in base_edges for v in e))
    
    # Build adjacency with edge indices
    adj = [[] for _ in range(base_n)]
    for idx, (u, v) in enumerate(base_edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Enumerate all non-backtracking closed walks of length 3 to g-1
    closed_walks = []
    
    for L in range(3, g):
        from collections import deque
        bfs_queue = deque()
        
        # BFS from each directed edge
        for start_u, start_v in base_edges:
            # Find the edge index for (start_u, start_v)
            start_eidx = None
            for idx, (u, v) in enumerate(base_edges):
                if (u == start_u and v == start_v) or (u == start_v and v == start_u):
                    start_eidx = idx
                    break
            if start_eidx is None:
                continue
            
            # Determine sign: if base_edges[start_eidx] = (start_u, start_v), sign=+1, else -1
            u0, v0 = base_edges[start_eidx]
            if u0 == start_u and v0 == start_v:
                start_sign = 1
            else:
                start_sign = -1
            
            # BFS: state is (prev, cur, path)
            # path is list of (eidx, sign)
            bfs_queue.append((start_u, start_v, [(start_eidx, start_sign)]))
            
            while bfs_queue:
                prev, cur, path = bfs_queue.popleft()
                if len(path) == L:
                    if cur == start_u:
                        # Closed walk!
                        closed_walks.append(path[:])
                    continue
                for (next_v, eidx) in adj[cur]:
                    if next_v == prev:
                        continue  # No backtracking
                    # Determine sign for this edge
                    u1, v1 = base_edges[eidx]
                    if u1 == cur and v1 == next_v:
                        sign = 1
                    else:
                        sign = -1
                    bfs_queue.append((cur, next_v, path + [(eidx, sign)]))
    
    if not closed_walks:
        return [0] * n_edges
    
    # Initialize voltages
    voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
    
    def count_violations(vols):
        count = 0
        for walk in closed_walks:
            s = 0
            for eidx, sign in walk:
                s = (s + sign * vols[eidx]) % m
            if s == 0:
                count += 1
        return count
    
    current_violations = count_violations(voltages)
    if current_violations == 0:
        return voltages[:]
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    while T > T_min and time.time() < deadline:
        e_idx = random.randint(0, n_edges - 1)
        old_v = voltages[e_idx]
        new_v = random.randint(0, m - 1)
        if new_v == old_v:
            continue
        voltages[e_idx] = new_v
        
        new_violations = count_violations(voltages)
        
        if new_violations <= current_violations:
            current_violations = new_violations
        else:
            voltages[e_idx] = old_v
        
        T *= cooling
        
        if current_violations == 0:
            return voltages[:]
    
    if current_violations == 0:
        return voltages[:]
    return None


def _build_lift(base_edges, voltages, m):
    """Build the lifted graph edges."""
    base_n = len(set(v for e in base_edges for v in e))
    edges = []
    
    for (u, v), vol in zip(base_edges, voltages):
        for s in range(m):
            su = u * m + s
            sv = v * m + (s + vol) % m
            if su < sv:
                edges.append((su, sv))
            else:
                edges.append((sv, su))
    
    return sorted(set(edges))


def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    
    # Check for duplicate edges
    if len(edges) != len(set(edges)):
        return False
        
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check using non-backtracking BFS from each edge
    INF = 10 ** 9
    for (u, v) in edges:
        # BFS from u, avoiding backtracking through the edge (u,v)
        # State: (prev, cur, distance)
        dist = {}
        q = []
        for w in adj[u]:
            if w == v:
                continue  # Don't start by going back through (u,v)
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
