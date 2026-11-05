import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Moore bound for k-regular girth g
    # For even g: n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2 - 1))
    # For odd g: n >= 1 + k * (1 + (k-1) + ... + (k-1)^((g-1)/2 - 1))
    # For k=4, g=10: even, so n >= 2 * (1 + 3 + 9 + 27 + 81) = 242
    # Known lower bound: 243
    
    if k != 4 or g != 10:
        # Fallback: use LZW for other parameters
        return _lzw_construct(k, g)
    
    # Try to find 4-regular graph with girth >= 10
    # Start from lower bound
    n_min = 243
    n_max = 320  # known upper bound
    
    best_n = None
    best_edges = None
    
    for n in range(n_min, min(n_max + 1, 300)):  # Try up to 300 first
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this n
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random 4-regular graph
            adj, valid = _random_4reg(n)
            if not valid:
                continue
            
            # Check girth
            if _has_girth_at_least(adj, n, g):
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                return best_n, best_edges
    
    # If we didn't find one, try larger n
    for n in range(300, n_max + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            adj, valid = _random_4reg(n)
            if not valid:
                continue
            
            if _has_girth_at_least(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: return LZW construction
    return _lzw_construct(k, g)


def _random_4reg(n):
    """Generate a random 4-regular simple graph using configuration model + 2-switches."""
    if time.time() > 145:
        return None, False
    
    # Configuration model: each vertex has 4 stubs
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair stubs
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    
    if not valid:
        return None, False
    
    # Convert to list
    adj = [list(s) for s in adj]
    
    # Do some 2-switches to randomize and potentially improve girth
    for _ in range(20):
        # Pick two random edges and swap endpoints
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) < 2:
            break
        
        e1, e2 = random.sample(edges, 2)
        (u, v), (w, x) = e1, e2
        
        # Check if swap is valid
        if u == w or u == x or v == w or v == x:
            continue
        if x in adj[u] or w in adj[v]:
            continue
        
        # Do the swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(w)
        adj[w].append(v)
    
    return adj, True


def _has_girth_at_least(adj, n, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] != dist[u] + 1:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _lzw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
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
