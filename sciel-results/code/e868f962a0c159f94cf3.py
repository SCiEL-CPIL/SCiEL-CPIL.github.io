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
    
    random.shuffle(list(range(n)))  # reseed
    
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

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
