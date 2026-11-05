import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Moore Bound
    if g == 1:
        return 2, [(0, 1)]
        
    if g % 2 == 0:
        # Even girth: Bipartite
        # Moore bound: 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        # Wait, standard Moore bound for even girth g is:
        # n >= 2 * ( (k-1)^{g/2} - 1 ) / (k-2)
        if k == 2:
            n = g
        else:
            n = 2 * ((k-1)**(g//2) - 1) // (k-2)
    else:
        # Odd girth:
        # n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        # n >= 1 + k * ( (k-1)^{(g-1)/2} - 1 ) / (k-2)
        if k == 2:
            n = g
        else:
            depth = (g - 1) // 2
            n = 1 + k * ((k-1)**depth - 1) // (k-2)
            
    n = max(n, 2*k) # Ensure n is at least 2k for k-regularity (trivial)
    
    # Try to find a graph with this n
    # If n is too small to be k-regular (n*k must be even), increment n
    if (n * k) % 2 != 0:
        n += 1

    best_n = n
    best_edges = None
    
    # Strategy: Randomized Configuration Model + Local Search
    # We will try to construct a graph with 'n' vertices.
    # If we fail to achieve girth g, we might need to increase n.
    # However, increasing n is expensive. We will try multiple random seeds for the same n first.
    
    max_restarts = 50
    attempts = 0
    
    while time.time() < deadline and attempts < max_restarts:
        attempts += 1
        
        # 1. Generate random k-regular graph
        edges = generate_random_k_regular(k, n)
        if not edges:
            # If we can't generate a valid k-regular graph (e.g. n too small), increase n
            n += 2 
            continue
            
        # 2. Check girth
        girth = calculate_girth(n, edges)
        
        if girth >= g:
            return n, edges
            
        # 3. Try to repair by rewiring
        # Find a short cycle and break it
        success = False
        for _ in range(100): # Limited repair attempts per graph
            cycle = find_shortest_cycle(n, edges)
            if not cycle or len(cycle) >= g:
                success = True
                break
                
            # Pick an edge in the cycle to rewire
            # cycle is a list of vertices [v0, v1, ..., v_{L-1}, v0]
            # Edges are (v0,v1), (v1,v2), ...
            # We want to remove (vi, vi+1) and add (vi, vj) and (vi+1, vk) such that degree is preserved
            # This is complex. Simpler: Just restart with a new random graph.
            # Or: Try to swap edges to break the specific cycle.
            
            # Let's just restart for now, it's simpler and often works for small n.
            # If n is close to Moore bound, random graphs are rare, so we might need to increase n.
            pass
            
        if success:
            return n, edges
            
        # If we failed to fix this random instance, maybe n is too small.
        # But random graphs with n=Moore bound are rare.
        # Let's increment n slightly to give the random search more space.
        # Incrementing by 2k ensures we can maintain k-regularity easily.
        n += 2
        
        # Update best found so far (in case we have a valid graph with higher n but better than infinite? No, we want min n)
        # We haven't found a valid one yet.
        
    # If we run out of time, we must return something valid.
    # We need to ensure we return a valid graph.
    # If we haven't found one, we should try a deterministic construction that is guaranteed to work,
    # even if n is larger.
    
    # Fallback: Lazebnik-Ustimenko-Woldar (from seed)
    # But we want to improve on it.
    # Let's use the seed's logic as a fallback, but ensure it's optimized.
    
    # Re-implement LUW quickly for fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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


def generate_random_k_regular(k, n):
    """Generates a random k-regular graph using the Configuration Model."""
    if n * k % 2 != 0:
        return None
    if k == 0:
        return []
    if n < k:
        return None
    # For small k, we can use a simpler method
    # Configuration model: 2 * n * k / 2 = n*k stubs
    # Each vertex has k stubs.
    # Randomly pair stubs.
    # Check for loops and multi-edges.
    
    # This is prone to loops. We will use a retry loop.
    for _ in range(10):
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
            
        if valid:
            return list(edges)
            
    return None

def calculate_girth(n, edges):
    """Calculates the girth of the graph using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        # Early exit if we found a cycle of length 2 or 3 (impossible for simple graph, but 3 is min)
        # Actually, BFS from one node finds the shortest cycle containing that node.
        # The global girth is the min over all nodes.
        
    return min_girth if min_girth != float('inf') else 2*n # Fallback

def find_shortest_cycle(n, edges):
    """Finds the shortest cycle in the graph."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_len = float('inf')
    min_cycle = None
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_len:
                        min_len = cycle_len
                        # Reconstruct cycle (optional, we just need to know it exists)
                        # For rewiring, we might need the actual cycle.
                        # But for now, we just restart.
                        
    return None if min_len == float('inf') else min_len # Return length or None

