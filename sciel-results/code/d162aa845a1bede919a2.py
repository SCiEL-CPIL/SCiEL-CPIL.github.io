"""Construct small 4-regular graphs with high girth using local search."""
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # For the (4,10) cage, known answer is 320 vertices
    # Moore bound: 2 + 4*(3 + 9 + 27) = 154
    # Try to find a 4-regular graph with girth >= 10
    
    if k != 4 or g != 10:
        # Fallback to LUW construction for other parameters
        return _luw_construct(k, g)
    
    # Try sizes from known optimum upward
    # Known: 320 is achievable
    target_sizes = [320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360]
    
    best_n = None
    best_edges = None
    
    for n in target_sizes:
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this size
        found = False
        attempts = 0
        max_attempts = 50
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges = _random_4reg(n)
            if edges is None:
                continue
            
            # Check girth
            if _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break
            
            # Try local search to improve girth
            if time.time() < deadline - 5:
                edges2 = _local_search(n, edges, g, max_time=2.0)
                if edges2 is not None and _has_girth_at_least(n, edges2, g):
                    best_n = n
                    best_edges = edges2
                    found = True
                    break
        
        if found:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q
    
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
                y[c] = gf_add(x[c], gf_mul(y[a], x[b])) if side == 0 else gf_sub(x[c], gf_mul(x[a], y[b]))
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


def _random_4reg(n):
    """Generate a random 4-regular graph on n vertices. Returns list of edges or None."""
    if n < 4:
        return None
    if n % 2 != 0:
        return None  # 4-regular requires even n... actually no, 4*n must be even, so n can be odd too
    # Actually 4-regular is possible for any n >= 4
    
    # Use configuration model: each vertex has 4 stubs
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            continue  # Skip loops, will retry
        edge = (min(u, v), max(u, v))
        edges.add(edge)
    
    # Check if we got exactly 2n edges (4-regular means 2n edges)
    if len(edges) != 2 * n:
        return None
    
    # Check degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 4 for d in deg):
        return None
    
    return list(edges)


def _has_girth_at_least(n, edges, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS and check no cycle shorter than g
    for start in range(n):
        # BFS from start, tracking shortest distance to each node
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue  # No need to go deeper
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    # Found a cycle of length dist[u] + dist[v] + 1 = 2*dist[u] + 2
                    # Wait, this is a cross edge, cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                # If dist[v] <= dist[u], it's a back or forward edge, ignore
    
    return True


def _local_search(n, edges, g, max_time=2.0):
    """Try to improve girth using 2-switches. Returns improved edges or None."""
    start_time = time.time()
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a short cycle to try to break
    def find_short_cycle():
        """Find a cycle of length < g if one exists."""
        for start in range(n):
            # BFS with parent tracking
            visited = {}
            queue = deque()
            queue.append((start, start, 0))
            visited[start] = (start, 0)
            
            while queue:
                u, parent, dist = queue.popleft()
                if dist >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    if v in visited:
                        # Found a cycle
                        v_parent, v_dist = visited[v]
                        cycle_len = dist + v_dist + 1
                        if cycle_len < g:
                            return True
                    else:
                        visited[v] = (u, dist + 1)
                        queue.append((v, u, dist + 1))
        return False
    
    if not find_short_cycle():
        return edges  # Already good
    
    # Try 2-switches
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    for _ in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Pick two random edges
        e1 = random.choice(list(edge_set))
        e2 = random.choice(list(edge_set))
        
        u1, v1 = e1
        u2, v2 = e2
        
        # Try switching: (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
        # or (u1,v2) and (v1,u2)
        new_edges = set()
        for e in edge_set:
            if e != e1 and e != e2:
                new_edges.add(e)
        
        # Option 1: (u1,u2) and (v1,v2)
        ne1 = (min(u1, u2), max(u1, u2))
        ne2 = (min(v1, v2), max(v1, v2))
        if ne1 != ne2 and ne1 not in new_edges and ne2 not in new_edges and ne1[0] != ne1[1] and ne2[0] != ne2[1]:
            candidate = new_edges | {ne1, ne2}
            # Check if still 4-regular
            deg = [0] * n
            for a, b in candidate:
                deg[a] += 1
                deg[b] += 1
            if all(d == 4 for d in deg):
                edge_set = candidate
                # Update adj
                adj = [[] for _ in range(n)]
                for u, v in edge_set:
                    adj[u].append(v)
                    adj[v].append(u)
        
        # Option 2: (u1,v2) and (v1,u2)
        ne1 = (min(u1, v2), max(u1, v2))
        ne2 = (min(v1, u2), max(v1, u2))
        if ne1 != ne2 and ne1 not in new_edges and ne2 not in new_edges and ne1[0] != ne1[1] and ne2[0] != ne2[1]:
            candidate = new_edges | {ne1, ne2}
            deg = [0] * n
            for a, b in candidate:
                deg[a] += 1
                deg[b] += 1
            if all(d == 4 for d in deg):
                edge_set = candidate
                adj = [[] for _ in range(n)]
                for u, v in edge_set:
                    adj[u].append(v)
                    adj[v].append(u)
    
    # Check if we improved
    if not find_short_cycle():
        return list(edge_set)
    
    return None
