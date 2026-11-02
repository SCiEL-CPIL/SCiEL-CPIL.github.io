import time
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    if k != 4 or g != 10:
        return _construct_lw(k, g)
    
    # Primary: LW graph construction (reliable algebraic method)
    lw_result = _construct_lw(k, g)
    if lw_result is not None:
        n_lw, edges_lw = lw_result
        if _verify_girth_fast(n_lw, edges_lw, g):
            # Try to improve with voltage graph search if time permits
            if time.time() - start_time < deadline - 30:
                vg_result = _voltage_graph_search(k, g, n_lw, start_time, deadline)
                if vg_result is not None:
                    n_vg, edges_vg = vg_result
                    if n_vg < n_lw:
                        return n_vg, edges_vg
            return n_lw, edges_lw
    
    # Fallback: try voltage graph search
    vg_result = _voltage_graph_search(k, g, float('inf'), start_time, deadline)
    if vg_result is not None:
        return vg_result
    
    # Last resort: random configuration model with verification
    return _random_search(k, g, start_time, deadline)


def _verify_girth_fast(n, edges, g):
    """Verify girth >= g using non-backtracking walks from every vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
    
    # Non-backtracking BFS from each vertex
    for start_v in range(n):
        visited = set()
        queue = deque()
        
        for nb in adj[start_v]:
            visited.add((nb, start_v))
            queue.append((nb, start_v, 1))
        
        while queue:
            curr, prev, depth = queue.popleft()
            
            if curr == start_v and depth < g:
                return False
            
            if depth >= g - 1:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, depth + 1))
    
    return True


def _construct_lw(k, g):
    """Lazebnik-Woldar graph construction."""
    q = k
    # d is the "girth parameter" for LW
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    
    if q == 4:
        # GF(4) arithmetic
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    # Rule for LW graph
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side, tuple(y))
    
    # BFS to enumerate all vertices
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges


def _voltage_graph_search(k, g, target_n, start_time, deadline):
    """Search for voltage graph over C4 with girth >= g."""
    best_n = target_n
    best_edges = None
    
    # Try m values: n = 4*m
    for m in range(40, min(81, (target_n // 4) + 1)):
        if time.time() - start_time > deadline - 5:
            break
        
        for attempt in range(30):
            if time.time() - start_time > deadline - 3:
                break
            
            v01 = random.randint(1, m - 1)
            v12 = random.randint(1, m - 1)
            v23 = random.randint(1, m - 1)
            v30 = random.randint(1, m - 1)
            
            # Avoid 4-cycle
            if (v01 + v12 + v23 + v30) % m == 0:
                continue
            
            n = 4 * m
            adj = [[] for _ in range(n)]
            edges = []
            
            for a in range(m):
                # (0,a) -- (1, a+v01)
                u1 = a
                v1 = m + (a + v01) % m
                edges.append((u1, v1))
                adj[u1].append(v1)
                adj[v1].append(u1)
                
                # (1,a) -- (2, a+v12)
                u2 = m + a
                v2 = 2*m + (a + v12) % m
                edges.append((u2, v2))
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                # (2,a) -- (3, a+v23)
                u3 = 2*m + a
                v3 = 3*m + (a + v23) % m
                edges.append((u3, v3))
                adj[u3].append(v3)
                adj[v3].append(u3)
                
                # (3,a) -- (0, a+v30)
                u4 = 3*m + a
                v4 = (a + v30) % m
                edges.append((u4, v4))
                adj[u4].append(v4)
                adj[v4].append(u4)
            
            # Check simplicity
            edge_set = set()
            is_simple = True
            for u, v in edges:
                if u == v:
                    is_simple = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    is_simple = False
                    break
                edge_set.add(key)
            
            if not is_simple:
                continue
            
            # Verify girth
            if _verify_girth_fast(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break
    
    if best_edges is not None:
        return best_n, best_edges
    return None


def _random_search(k, g, start_time, deadline):
    """Random configuration model search as last resort."""
    best_n = float('inf')
    best_edges = None
    
    for n in range(244, 400, 4):  # Even n for 4-regular
        if time.time() - start_time > deadline - 10:
            break
        
        for attempt in range(200):
            if time.time() - start_time > deadline - 5:
                break
            
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            edges = []
            used = set()
            valid = True
            
            for i in range(0, 2 * n * k, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in used:
                    valid = False
                    break
                used.add(key)
                edges.append((u, v))
            
            if not valid:
                continue
            
            # Quick check: no 4-cycles
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            has_4cycle = False
            for u in range(n):
                for i in range(len(adj[u])):
                    a = adj[u][i]
                    for j in range(i + 1, len(adj[u])):
                        b = adj[u][j]
                        if set(adj[a]) & set(adj[b]):
                            has_4cycle = True
                            break
                    if has_4cycle:
                        break
                if has_4cycle:
                    break
            
            if has_4cycle:
                continue
            
            # Full girth check
            if _verify_girth_fast(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break
    
    if best_edges is not None:
        return best_n, best_edges
    return None
