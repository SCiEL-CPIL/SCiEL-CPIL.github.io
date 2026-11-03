import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for k=3, g=16 is 765.
    # n must be even.
    # We aim for n close to 765.
    
    best_n = float('inf')
    best_edges = []
    
    # Candidate sizes
    # 766, 768, 770...
    # Let's try a few sizes to see if we can get a valid graph.
    # Smaller is better.
    
    # We'll try sizes in increasing order.
    # If we find a valid graph for n=766, that's optimal (or close to it).
    # If not, try 768, etc.
    
    # Max n to try before falling back to LUW
    max_n_to_try = 900 
    
    n = 766
    while n <= max_n_to_try and time.time() < deadline:
        
        # Try multiple random initializations for this n
        found = False
        attempts = 0
        # Increase attempts for smaller n as they are harder?
        # Or just fixed attempts per n.
        max_attempts = 100 
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges, adj = generate_random_cubic(n)
            
            if not edges:
                continue
                
            # Check girth
            if has_girth_at_least(adj, n, g):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
        n += 2 # n must be even
        
    if best_n == float('inf'):
        # Fallback: Use the LUW construction from the seed if we run out of time or fail.
        # But the seed code is in the prompt. We can copy it.
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

    return best_n, best_edges

def generate_random_cubic(n):
    """Generates a random 3-regular graph on n vertices."""
    if n % 2 != 0:
        return [], []
    if n < 4:
        return [], []
        
    # Use a pairing of stubs
    stubs = [i for i in range(n) for _ in range(3)]
    
    # Try a few times to get a simple graph
    for _ in range(20):
        random.shuffle(stubs)
        edges = []
        adj = [[] for _ in range(n)]
        edge_set = set()
        
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        if valid:
            return edges, adj
            
    return [], []

def has_girth_at_least(adj, n, g):
    """Checks if graph has girth at least g."""
    limit = g // 2 - 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= limit:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True
