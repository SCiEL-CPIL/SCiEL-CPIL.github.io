<<<<<<< SEARCH
import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Base graph: Petersen (10 vertices, 15 edges)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1 from every vertex
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute coefficient vectors
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_coeffs.append(nonzero)
    
    # Moore bound for n(3,16) is 512, so n = 10*m >= 512 means m >= 52
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    for m in range(52, 2000):
        if time.time() > deadline - 10.0:
            break
        
        candidate_n = 10 * m
        
        time_left = deadline - time.time()
        max_tries = min(3000, max(300, int(time_left / 0.0003)))
        
        found = False
        for trial in range(max_tries):
            if time.time() > deadline - 5.0:
                break
            
            volts = [random.randint(1, m - 1) for _ in range(len(edges))]
            
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            
            valid = True
            for nonzero in walk_coeffs:
                s = 0
                for idx, c in nonzero:
                    s += c * volts[idx]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            best_m = m
            best_n = candidate_n
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))
            best_edges = lift_edges
            break
    
    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges
=======
import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Moore bound for n(3,16) is 512
    moore_bound = 512
    
    best_n = float('inf')
    best_edges = None
    best_girth = 0
    
    # Try multiple starting sizes, from small to large
    for n in range(moore_bound, 2000, 16):
        if time.time() > deadline - 20.0:
            break
        
        # Skip if n is too close to previous best
        if n >= best_n:
            break
        
        # Generate a random 3-regular graph on n vertices
        random.seed(n * 7919 + 12345)
        adj = _random_3_regular(n)
        if adj is None:
            continue
        
        # Iteratively increase girth via 2-switches
        girth = _compute_girth(adj, n)
        
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break
            if girth >= g:
                break
            
            # Find a short cycle and perform a 2-switch to eliminate it
            switched = False
            for _attempt in range(200):
                if time.time() > deadline - 15.0:
                    break
                
                # Find a random cycle of length < g
                cycle = _find_short_cycle(adj, n, g)
                if cycle is None:
                    break
                
                # Try to perform a 2-switch on this cycle
                if _try_2switch(adj, n, cycle):
                    switched = True
                    girth = _compute_girth(adj, n)
                    break
            
            if not switched:
                # No more switches possible; try another random graph
                break
        
        if girth >= g:
            # Convert adjacency to edge list
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            # Verify the graph
            if _verify_graph(n, edges, k, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid graph, stop searching
    
    if best_edges is not None:
        return best_n, best_edges
    
    return _luw_fallback(k, g)

def _random_3_regular(n):
    """Generate a random 3-regular simple graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    
    random.shuffle(range(n))  # reseed
    
    # Configuration model: each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    # Randomly pair up stubs
    random.shuffle(stubs)
    edges_set = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # loop, retry
        key = (min(u, v), max(u, v))
        if key in edges_set:
            return None  # multi-edge, retry
        edges_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    return adj

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g using BFS from a random vertex."""
    for start in range(n):
        if len(adj[start]) < 3:
            continue
        
        # BFS to find shortest cycle back to start
        visited = {start: (-1, 0)}  # vertex -> (parent, distance)
        queue = deque([(start, -1, 0)])
        
        while queue:
            curr, prev, dist = queue.popleft()
            if dist >= g - 1:
                break
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if nb == start:
                    if dist + 1 < g:
                        # Reconstruct cycle
                        cycle = [curr, start]
                        p = curr
                        while p != -1:
                            p, _ = visited[p] if p in visited else (-1, 0)
                            if p != -1 and p != start:
                                cycle.append(p)
                        return cycle
                elif nb not in visited:
                    visited[nb] = (curr, dist + 1)
                    queue.append((nb, curr, dist + 1))
    
    return None

def _try_2switch(adj, n, cycle):
    """Try to perform a 2-switch to break a cycle. Returns True if successful."""
    if len(cycle) < 4:
        return False
    
    # A 2-switch takes two edges (a,b) and (c,d) and replaces with (a,c) and (b,d)
    # or (a,d) and (b,c), provided the new edges don't create loops or multi-edges
    
    # Pick two non-adjacent edges on the cycle
    m = len(cycle)
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        for j in range(i + 2, m):
            if j == (i + 1) % m:
                continue
            c = cycle[j]
            d = cycle[(j + 1) % m]
            
            # Check that a,b and c,d are not adjacent on the cycle
            if abs(i - j) <= 1 or abs(i - j) >= m - 1:
                continue
            
            # Try switch: remove (a,b) and (c,d), add (a,c) and (b,d)
            if a == c or a == d or b == c or b == d:
                continue
            
            # Check no existing edges
            if c in adj[a] or d in adj[b]:
                continue
            
            # Perform the switch
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            
            return True
    
    return False

def _compute_girth(adj, n):
    """Compute the girth (shortest cycle length) of a graph."""
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start to find shortest cycle back to start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] >= min_girth - 1:
                continue
            for nb in adj[curr]:
                if nb == parent[curr]:
                    continue
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    queue.append(nb)
                elif dist[nb] <= dist[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    return min_girth if min_girth < float('inf') else n

def _verify_graph(n, edges, k, g):
    """Verify that the graph is k-regular, simple, and has girth >= g."""
    # Check degree
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        deg[u] += 1
        deg[v] += 1
    
    for d in deg:
        if d != k:
            return False
    
    # Check girth using non-backtracking walks
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS with non-backtracking constraint
        # State: (current_vertex, previous_vertex, length)
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            curr, prev, length = queue.popleft()
            if length >= g:
                continue
            if curr == start and length >= 3:
                return False  # Found cycle shorter than g
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, length + 1))
    
    return True
>>>>>>> REPLACE

