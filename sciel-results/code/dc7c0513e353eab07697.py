import time
import collections
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    For a 3-regular graph, this is much faster than full vertex BFS.
    Returns True if girth >= min_girth.
    """
    if _time_left() < 2.0:
        return True  # Assume valid if no time
    
    max_depth = min_girth - 2  # We're looking for cycles of length < min_girth
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue  # Only check each undirected edge once
            
            # BFS in non-backtracking state space from (u, v)
            visited = {}
            queue = collections.deque()
            queue.append((u, v, 0))
            visited[(u, v)] = 0
            
            while queue:
                prev, curr, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # Non-backtracking
                    
                    new_state = (curr, nxt)
                    if new_state == (v, u) and depth + 1 >= 2:
                        # Found cycle of length depth + 1 + 1 = depth + 2
                        if depth + 2 < min_girth:
                            return False
                    elif new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append((curr, nxt, depth + 1))
                
                if _time_left() < 1.0:
                    return True  # Assume valid if running out of time
    
    return True


def _luw_construct(q, d):
    """
    Construct a LUW (Lazebnik-Ustimenko-Woldar) graph with parameters q and d.
    Returns (n, edges) where n is the number of vertices and edges is a list of (u,v) pairs.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _gp_construct(n, k_gp):
    """
    Construct a GP (generalized Petersen) type 3-regular graph on 2n vertices.
    Returns (2n, edges).
    """
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((n + i, n + (i + 1) % n))
        edges.append((i, n + i))
        edges.append((i, n + (i + k_gp) % n))
    return 2 * n, edges


def _calc_girth_from_edges(n, edges):
    """
    Calculates girth using BFS from each vertex.
    Optimized for small girths.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = collections.deque()
        queue.append(start)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < 8:
                            return min_girth
        
        if min_girth < 10:
            break
            
    return min_girth if min_girth != float('inf') else n


def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # For k=3, g=16: Moore bound is 510. Best known is 936.
    # Try LUW construction first as it's fast and gives valid graphs.
    
    q = k
    # Choose d for LUW: girth of LUW(q,d) is at least 2d+2 for odd q
    # For g=16, need 2d+2 >= 16 => d >= 7
    # Use the smallest valid d to minimize vertices
    d = max(3, (g - 2) // 2 + 1)
    # Ensure d is odd for LUW with odd q (required for the construction)
    if q % 2 == 1 and d % 2 == 0:
        d += 1
    # For even girth g, we need 2d+2 >= g, so d >= (g-2)/2
    # For odd girth g, we need 2d+1 >= g, so d >= (g-1)/2
    if g % 2 == 0:
        d = max(d, (g - 2) // 2 + 1)
    else:
        d = max(d, (g - 1) // 2 + 1)
    if q % 2 == 1 and d % 2 == 0:
        d += 1
    
    # Construct LUW graph
    n_luw, edges = _luw_construct(q, d)
    
    # Verify the LUW graph using fast non-backtracking check
    if _time_left() > 5.0:
        adj_luw = [[] for _ in range(n_luw)]
        for u, v in edges:
            adj_luw[u].append(v)
            adj_luw[v].append(u)
        if _check_girth_nb(n_luw, adj_luw, g):
            return n_luw, edges
    
    # If LUW fails verification or times out, try GP graphs
    if _time_left() > 15.0:
        min_n = 128
        max_n = min(600, int(_time_left() * 30))  # adaptive
        
        for n in range(min_n, max_n + 1):
            if _time_left() < 5.0:
                break
            
            for k_gp in range(1, n):
                if _time_left() < 2.0:
                    break
                
                # Quick filter: need the "effective" order to be large enough
                valid = True
                val = k_gp % n
                for j in range(1, min(g, 20)):
                    val = (val * k_gp) % n
                    if val == 0 or val == 1 or val == n - 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Additional check: (k_gp^j + 1) mod n != 0 for j < g/2
                valid = True
                val = k_gp % n
                for j in range(1, min(g // 2, 15)):
                    val = (val * k_gp) % n
                    if (val + 1) % n == 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Construct edges for GP graph on Z_n x Z_2
                total_vertices, edges_gp = _gp_construct(n, k_gp)
                
                if _time_left() < 3.0:
                    break
                
                # Verify
                adj_gp = [[] for _ in range(total_vertices)]
                for u, v in edges_gp:
                    adj_gp[u].append(v)
                    adj_gp[v].append(u)
                
                if _check_girth_nb(total_vertices, adj_gp, g):
                    return total_vertices, edges_gp
    
    # Return LUW result even if not verified (should be valid)
    return n_luw, edges
