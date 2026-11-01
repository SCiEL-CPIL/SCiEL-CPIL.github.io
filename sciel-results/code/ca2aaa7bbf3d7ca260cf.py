import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    New Approach: Circulant Graph Search for k=3, g=16.
    Falls back to LUW construction.
    """
    if k == 3 and g == 16:
        # Get the LUW graph as a fallback and to determine search upper bound
        n_luw, edges_luw = luw_construct(3, 16)
        
        start_time = time.time()
        timeout = 100.0
        
        # Search for circulant graphs with N < n_luw
        # Limit search to reasonable N to avoid timeout
        max_search_n = min(n_luw - 2, 5000)
        
        for N in range(512, max_search_n, 2):
            if time.time() - start_time > timeout:
                break
            
            # Try random shifts s1 for circulant graph C_N({s1, N-s1, N/2})
            for _ in range(50):
                s1 = random.randint(1, N//2 - 1)
                edges = generate_circulant_3_reg(N, s1)
                if edges is None:
                    continue
                
                girth = compute_girth(N, edges, min_girth_target=16)
                if girth >= 16:
                    return N, edges

        return n_luw, edges_luw
    else:
        return luw_construct(k, g)

def generate_circulant_3_reg(N, s1):
    """
    Generates a 3-regular circulant graph on N vertices.
    Edges: (i, i+s1), (i, i-s1), (i, i+N/2)
    """
    if N % 2 != 0:
        return None
    if s1 == 0 or s1 == N//2:
        return None
    
    edges = []
    seen = set()
    half = N // 2
    
    for i in range(N):
        # Edge 1: i to i+s1
        j1 = (i + s1) % N
        if i < j1:
            key = (i, j1)
            if key not in seen:
                seen.add(key)
                edges.append(key)
        
        # Edge 2: i to i-s1
        j2 = (i - s1) % N
        if i < j2:
            key = (i, j2)
            if key not in seen:
                seen.add(key)
                edges.append(key)
                
        # Edge 3: i to i+half
        j3 = (i + half) % N
        if i < j3:
            key = (i, j3)
            if key not in seen:
                seen.add(key)
                edges.append(key)
                
    # Verify degree
    deg = [0] * N
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        
    if any(d != 3 for d in deg):
        return None
        
    return edges

def generate_random_3_regular(N):
    """Generates a random 3-regular graph on N vertices."""
    if N % 2 != 0:
        return None
    
    # Use the configuration model
    # Create a list of stubs
    stubs = []
    for i in range(N):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = []
    seen = set()
    for i in range(0, N * 3, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop, reject
        if (u, v) in seen or (v, u) in seen:
            return None # Multi-edge, reject
        seen.add((u, v))
        seen.add((v, u))
        edges.append((u, v))
        
    return edges

def compute_girth(N, edges, min_girth_target=None):
    """Computes the girth of the graph using BFS with early exit."""
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(N):
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth_target and min_girth < min_girth_target:
                            return min_girth
        
    return min_girth if min_girth != float('inf') else N

def local_repair(N, edges, target_girth, time_limit):
    """
    Attempts to increase the girth by breaking short cycles via 2-switches.
    Returns new edges if successful, else None.
    """
    start_time = time.time()
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Simple heuristic: Find a short cycle and break it.
    # This is a simplified version. A full implementation would be complex.
    # Given time constraints, we might skip this if it's too slow to implement correctly.
    # For now, return None to indicate failure to repair.
    return None

def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    """
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
