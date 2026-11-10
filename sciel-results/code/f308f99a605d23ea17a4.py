import time
import random
from collections import deque

def _luw_construct(d, q):
    """
    Construct Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges).
    """
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

def _check_girth_nonbacktracking(n, edges, g):
    """
    Check girth >= g using non-backtracking walks from every vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = g - 1
    
    for start in range(n):
        visited = {}
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            if state not in visited:
                visited[state] = 1
                queue.append((nb, start, 1))
        
        while queue:
            curr, prev, dist = queue.popleft()
            
            if curr == start:
                if dist < g:
                    return False
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                new_state = (nb, curr)
                new_dist = dist + 1
                if new_state not in visited:
                    visited[new_state] = new_dist
                    if new_dist < g:
                        queue.append((nb, curr, new_dist))
    
    return True

def _construct_cage_4_10_v3():
    """
    Attempt to construct a (4,10)-cage on 320 vertices using local search.
    """
    n = 320
    half = 160
    # Bipartite graph: Left 0..159, Right 160..319
    # Each left node connects to 4 right nodes.
    # Each right node connects to 4 left nodes.
    
    # Initialize random matching
    # left[i] is a list of 4 right neighbors
    left_neigh = [random.sample(range(half), 4) for _ in range(half)]
    
    # Check if this is a valid 4-regular graph (it is by construction of sampling, 
    # but we need to ensure right nodes also have degree 4? 
    # No, random sampling of left neighbors does NOT guarantee right degrees are 4.
    # We need a random 4-regular bipartite graph.
    
    # Use configuration model or random regular graph generation.
    # Simple method: Start with random permutation and shuffle.
    # Or use networkx? No, standard lib only.
    
    # Let's use a simple random walk / swap algorithm to generate a random 4-regular bipartite graph.
    # 1. Start with a valid 4-regular bipartite graph (e.g., circulant).
    # 2. Randomly swap edges to randomize.
    
    # Initial circulant: left i connects to right (i+j) % 160 for j in 0,1,2,3.
    left_neigh = [[(i + j) % half for j in range(4)] for i in range(half)]
    
    # Randomize by swapping
    for _ in range(10000):
        i1 = random.randint(0, half-1)
        i2 = random.randint(0, half-1)
        if i1 == i2: continue
        j1 = random.randint(0, 3)
        j2 = random.randint(0, 3)
        v1 = left_neigh[i1][j1]
        v2 = left_neigh[i2][j2]
        if v1 == v2: continue
        # Swap
        left_neigh[i1][j1] = v2
        left_neigh[i2][j2] = v1
    
    # Convert to edge list
    edges = []
    for i in range(half):
        for j in left_neigh[i]:
            edges.append((i, j + half))
            
    return n, edges

def _find_short_cycle(adj, n, max_len=8):
    """
    Find a cycle of length <= max_len.
    Returns list of vertices or None.
    """
    for start in range(n):
        # BFS up to depth max_len/2
        visited = {start: []}
        queue = deque([(start, 0, [start])])
        while queue:
            curr, dist, path = queue.popleft()
            if dist >= max_len // 2:
                continue
            for nb in adj[curr]:
                if nb == start and dist + 1 >= 4: # Cycle found (len >= 4)
                    # Check if it's a simple cycle (no repeats in path)
                    if len(set(path + [start])) == len(path) + 1:
                        return path + [start]
                if nb not in visited:
                    visited[nb] = dist + 1
                    queue.append((nb, dist + 1, path + [nb]))
                elif nb == start:
                    pass
                # Note: This BFS doesn't strictly prevent backtracking in the path check 
                # but the 'simple cycle' check handles it.
                # For efficiency, we can just check for any cycle.
    return None

def _construct_cage_4_10_v4():
    """
    Local search for (4,10)-cage.
    """
    n = 320
    half = 160
    deadline = time.time() + 100
    
    # Init
    left_neigh = [[(i + j) % half for j in range(4)] for i in range(half)]
    
    # Randomize
    for _ in range(20000):
        i1 = random.randint(0, half-1)
        i2 = random.randint(0, half-1)
        if i1 == i2: continue
        j1 = random.randint(0, 3)
        j2 = random.randint(0, 3)
        v1 = left_neigh[i1][j1]
        v2 = left_neigh[i2][j2]
        if v1 == v2: continue
        left_neigh[i1][j1] = v2
        left_neigh[i2][j2] = v1
        
    def get_edges():
        edges = []
        for i in range(half):
            for j in left_neigh[i]:
                edges.append((i, j + half))
        return edges

    def build_adj(edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Check initial
    edges = get_edges()
    adj = build_adj(edges)
    
    while time.time() < deadline:
        # Check for cycles of length 4, 6, 8
        # Use BFS from each node up to depth 4
        found_cycle = None
        for start in range(n):
            visited = {start: (0, -1)} # dist, prev
            queue = deque([(start, 0, -1)])
            while queue:
                curr, dist, prev = queue.popleft()
                if dist >= 4:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if nb == start and dist + 1 >= 4:
                        # Found cycle
                        # Reconstruct path? 
                        # For simplicity, just return a signal to swap
                        found_cycle = True
                        break
                    if nb not in visited:
                        visited[nb] = (dist + 1, curr)
                        queue.append((nb, dist + 1, curr))
                if found_cycle:
                    break
            if found_cycle:
                break
        
        if not found_cycle:
            return n, edges
            
        # Perform a random swap to break cycles
        i1 = random.randint(0, half-1)
        i2 = random.randint(0, half-1)
        if i1 == i2:
            continue
        j1 = random.randint(0, 3)
        j2 = random.randint(0, 3)
        v1 = left_neigh[i1][j1]
        v2 = left_neigh[i2][j2]
        if v1 == v2:
            continue
        left_neigh[i1][j1] = v2
        left_neigh[i2][j2] = v1
        edges = get_edges()
        adj = build_adj(edges)
        
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = None
    best_edges = None
    
    if k == 4 and g == 10:
        # Try local search for 320
        try:
            n, edges = _construct_cage_4_10_v4()
            if _check_girth_nonbacktracking(n, edges, g):
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == k for d in deg):
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
            
        if best_n is None:
            # Fallback to LUW
            try:
                n, edges = _luw_construct(4, 4)
                if _check_girth_nonbacktracking(n, edges, g):
                    best_n = n
                    best_edges = edges
            except Exception:
                pass
                
    if best_n is None:
        # Generic fallback
        d = max(3, (g - 2) // 2)
        try:
            n, edges = _luw_construct(d, k)
            best_n = n
            best_edges = edges
        except Exception:
            best_n = k + 1
            best_edges = []
            for i in range(k):
                best_edges.append((0, i + 1))
                
    return best_n, best_edges
