import time
import random
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _greedy_construct(k, g, n, time_limit=5.0):
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    active = list(range(n))
    random.shuffle(active)
    
    # We need to add k * n / 2 edges
    # active contains vertices with degree < k
    
    while active:
        if time.time() - start_time > time_limit:
            return None
            
        # Pick a random vertex from active
        u_idx = random.randint(0, len(active) - 1)
        u = active[u_idx]
        
        # Find candidates for v
        # v must be in active, v != u, and dist(u, v) >= g - 1
        
        # BFS from u up to depth g - 2
        # If dist(u, v) < g - 1, v is forbidden
        forbidden = set()
        queue = deque([(u, 0)])
        forbidden.add(u)
        while queue:
            node, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for neighbor in adjacency[node]:
                if neighbor not in forbidden:
                    forbidden.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # Candidates are in active but not in forbidden
        # Also, we should probably prioritize vertices with higher degree?
        # Or just random.
        
        # To speed up, we can iterate through active and check
        candidates = []
        for v in active:
            if v in forbidden:
                continue
            # v is a candidate
            candidates.append(v)
            
        if not candidates:
            return None
            
        v = random.choice(candidates)
        
        # Add edge
        adjacency[u].add(v)
        adjacency[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
        
        # Update active
        # Remove u if degree == k
        # Remove v if degree == k
        # This is O(n). We can do it lazily or rebuild.
        # Rebuilding is O(n). n is small (~1000).
        # But we do this for every edge. Edges ~ 2500. 2500 * 1000 = 2.5M ops. OK.
        
        new_active = []
        for node in active:
            if degrees[node] < k:
                new_active.append(node)
        active = new_active
        
    # Verify
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
            
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges

def construct(k, g):
    # Fallback to LUW
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try greedy for k=5, g=9
    if k == 5 and g == 9:
        # Try increasing n
        # Start from Moore bound approx 428
        # Best known 1116
        # LUW is 6250
        # Try 500, 600, ..., 2000
        # Time budget: 120s total.
        # LUW construction is fast.
        # Greedy attempts: 5s each.
        # We can try ~20 sizes.
        
        start_time = time.time()
        for n in range(500, 3000, 100):
            if time.time() - start_time > 110:
                break
            # Try a few seeds for this n
            for seed in range(5):
                random.seed(seed)
                res = _greedy_construct(k, g, n, time_limit=2.0)
                if res is not None:
                    n_found, edges_found = res
                    if n_found < best_n:
                        best_n = n_found
                        best_edges = edges_found
                    # If we found a small one, we can stop or try smaller?
                    # We are iterating n upwards. So the first success is the smallest for this n.
                    # But we might fail for n=500 and succeed for n=600.
                    # We want the minimum n.
                    # So we should break out of the seed loop and continue to next n?
                    # No, if we succeed for n=600, we have a graph of size 600.
                    # We should check if we can do better with n=500? We already tried.
                    # So we can return immediately?
                    # No, maybe n=550 would work?
                    # But we are stepping by 100.
                    # Let's just return the first success found in the increasing order.
                    # Wait, if we find n=600, it's better than LUW (6250).
                    # But maybe n=500 works with a different seed?
                    # We tried 5 seeds.
                    # Let's return the first success found.
                    return best_n, best_edges
            
    return best_n, best_edges
