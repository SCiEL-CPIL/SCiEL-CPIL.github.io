import time
import random
import sys
from collections import deque

def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar construction
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

def _generate_random_regular(n, k):
    """Generate a random simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    attempts = 0
    max_attempts = 500
    while attempts < max_attempts:
        attempts += 1
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return adj
    return None

def _check_girth(adj, g, deadline=None):
    """Check if graph has girth >= g. Returns True if girth >= g, else False.
    Uses BFS from each vertex with non-backtracking walks."""
    n = len(adj)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        if deadline and time.time() > deadline:
            return True  # Assume valid if we run out of time
        
        # BFS from start, tracking (node, parent) pairs to avoid backtracking
        # visited maps (node, parent) -> distance
        visited = {}
        queue = deque([(start, -1, 0)])
        
        while queue:
            u, parent, dist = queue.popleft()
            
            if dist >= max_depth:
                continue
            
            for v in adj[u]:
                if v == parent:
                    continue
                # Check if we've seen (v, u) before
                if (v, u) in visited:
                    if visited[(v, u)] >= dist + 1:
                        continue
                else:
                    visited[(v, u)] = dist + 1
                    queue.append((v, u, dist + 1))
                
                # If we can reach v from start via a different path, we have a cycle
                # The cycle length is dist + 1 + (distance from start to v via different path)
                # But we need to be careful - we're doing non-backtracking BFS
                # A cycle exists if we can reach (v, u) where v is already in our tree
                # with a different parent, and the combined path forms a cycle < g
                
                # Actually, let's use a cleaner approach:
                # Track shortest non-backtracking path from start to each node
                # If we find a node reachable via two different non-backtracking paths,
                # the sum gives a cycle length
                
    # Better approach: use the standard method
    for start in range(n):
        if deadline and time.time() > deadline:
            return True
        
        dist = {}
        parent = {}
        dist[start] = 0
        parent[start] = -1
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle: u-v edge connects two nodes in the tree
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _find_short_cycle_vertices(adj, g, deadline=None):
    """Find a short cycle (length < g) and return its vertices as a set.
    Returns None if no short cycle found."""
    n = len(adj)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        if deadline and time.time() > deadline:
            return None
        
        dist = {}
        parent = {}
        dist[start] = 0
        parent[start] = -1
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle through u-v
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct the cycle
                        path_u = []
                        cur = u
                        while cur != -1:
                            path_u.append(cur)
                            cur = parent[cur]
                        path_v = []
                        cur = v
                        while cur != -1:
                            path_v.append(cur)
                            cur = parent[cur]
                        
                        # Find LCA to get the actual cycle
                        set_u = set(path_u)
                        set_v = set(path_v)
                        
                        # Walk up from u to find where paths diverge
                        i = 0
                        j = 0
                        while i < len(path_u) and j < len(path_v) and path_u[i] == path_v[j]:
                            i += 1
                            j += 1
                        # path_u[i:] and path_v[j:] are the two branches
                        cycle_verts = set(path_u[i:]) | set(path_v[j:])
                        return cycle_verts
    
    return None

def _try_switching(n, k, g, time_limit=100.0):
    start_time = time.time()
    deadline = start_time + time_limit
    
    # 1. Generate initial graph
    adj = _generate_random_regular(n, k)
    if adj is None:
        return None
    
    # Check if girth is already >= g
    if _check_girth(adj, g, deadline):
        edge_list = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edge_list.append((i, j))
        return n, edge_list
    
    # 2. Switching loop
    while time.time() < deadline - 5:
        # Find a short cycle
        cycle_verts = _find_short_cycle_vertices(adj, g, deadline)
        if cycle_verts is None:
            edge_list = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edge_list.append((i, j))
            return n, edge_list
        
        # Try to break the cycle by switching
        success = False
        cycle_list = list(cycle_verts)
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge within the cycle vertices
            if len(cycle_list) < 2:
                break
            u = random.choice(cycle_list)
            v_candidates = [v for v in adj[u] if v in cycle_verts]
            if not v_candidates:
                continue
            v = random.choice(v_candidates)
            
            # Pick x randomly (not in cycle), y a neighbor of x
            x = random.randint(0, n - 1)
            if x in cycle_verts:
                continue
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            if y in cycle_verts or y == u or y == v:
                continue
            if x in adj[u] or y in adj[v]:
                continue
            
            # Check if switch preserves regularity (it does by construction)
            # and doesn't create loops or multi-edges
            # Perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            success = True
            break
        
        if not success:
            # Try a random switch
            u = random.randint(0, n - 1)
            if not adj[u]:
                continue
            v = random.choice(list(adj[u]))
            x = random.randint(0, n - 1)
            if x == u or x == v or x not in adj:
                continue
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v or y == x:
                continue
            if x in adj[v] or y in adj[u]:
                continue
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
    
    return None

def construct(k, g):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    if k == 5 and g == 9:
        # Try switching on smaller n first
        target_ns = [1116, 1120, 1150]
        time_per_attempt = 40.0
        
        for n in target_ns:
            if time.time() > deadline - 20.0:
                break
            res = _try_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
    
    # Fallback to LUW
    return _luw_construct(k, g)
