import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore bound
    if g % 2 == 0:
        # Even girth
        # n >= 2 * (k * (k-1)^(g/2 - 1) - 1) / (k-1) ? 
        # Standard Moore bound for even girth g:
        # n >= 2 * (1 + k * sum_{i=0}^{g/2 - 2} (k-1)^i)
        # Sum is ( (k-1)^(g/2 - 1) - 1 ) / (k - 2)
        # Wait, the formula is n >= 2 * (k * (k-1)^(g/2 - 1) - 1) / (k - 2) is not quite right.
        # Let's use the standard definition:
        # For even g, n >= 2 * (1 + k * ((k-1)^(g/2 - 1) - 1)/(k-2))
        # For odd g, n >= 1 + k * ((k-1)^( (g-1)/2 ) - 1)/(k-2)
        
        half_g = g // 2
        # n >= 2 * (1 + k * sum_{i=0}^{half_g - 2} (k-1)^i)
        # sum = 1 + (k-1) + ... + (k-1)^(half_g-2)
        # This is a geometric series.
        if k == 2:
            n_min = g
        else:
            # 1 + k * ( ((k-1)^(half_g-1) - 1) / (k-2) )
            # Actually, the number of vertices in a Moore graph of even girth g is:
            # 2 * (k * (k-1)^(g/2 - 1) - 1) / (k - 2)
            # Let's verify for k=3, g=4 (C4): 2 * (3*1 - 1) / 1 = 4. Correct.
            # k=3, g=6 (K3,3): 2 * (3*2 - 1) / 1 = 10. Correct.
            # k=5, g=4: 2 * (5*1 - 1) / 3 = 8/3 -> 3? No, C5 is k=2.
            # For k=5, g=4, min n is 5 (C5 is girth 5).
            # The formula gives 2 * (5 * 1 - 1) / 3 = 8/3 = 2.66 -> 3.
            # Wait, the Moore bound is a lower bound.
            # Let's just use a safe lower bound.
            n_min = int(2 * (k * ((k-1) ** (half_g - 1) - 1) / (k - 2)) + 1)
            # Actually, for k=5, g=9 (odd):
            # n >= 1 + k * ((k-1)^((g-1)/2) - 1) / (k-2)
            # (g-1)/2 = 4.
            # n >= 1 + 5 * (4^4 - 1) / 3 = 1 + 5 * 255 / 3 = 1 + 5 * 85 = 426.
            # So 428 is the cited lower bound.
    else:
        # Odd girth
        half_g = (g - 1) // 2
        if k == 2:
            n_min = g
        else:
            # n >= 1 + k * sum_{i=0}^{half_g - 1} (k-1)^i
            # sum = ((k-1)^half_g - 1) / (k-2)
            n_min = int(1 + k * (((k-1) ** half_g - 1) / (k - 2)) + 1)
            
    # We will try to find a graph with n >= n_min
    # Start with n_min and go up.
    # Randomized Greedy
    
    def try_random_greedy(n):
        # Initialize adjacency list
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        
        # Helper to check if adding edge (u, v) creates a cycle of length < g
        # We can do a BFS/DFS from u and v to ensure distance >= g/2
        # Actually, we need to ensure that the shortest path between u and v is at least g-1?
        # No, if we add edge (u, v), we create a cycle of length L = dist(u, v) + 1.
        # We need L >= g. So dist(u, v) >= g - 1.
        
        def min_dist(u, v, max_dist):
            # BFS from u to v, stop if distance > max_dist
            if u == v: return 0
            visited = {u: 0}
            queue = [u]
            while queue:
                next_queue = []
                for node in queue:
                    for neighbor in adj[node]:
                        if neighbor == v:
                            return visited[node] + 1
                        if neighbor not in visited:
                            visited[neighbor] = visited[node] + 1
                            if visited[neighbor] < max_dist:
                                next_queue.append(neighbor)
                queue = next_queue
            return max_dist + 1 # Or infinity
        
        # To speed up, we can precompute distances or use a simpler check.
        # For small n, BFS is fine.
        
        # Random order of vertices
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # We need to add k*n/2 edges.
        # Greedy: pick a vertex with min degree, pick a neighbor with min degree, check girth.
        
        for _ in range(k * n // 2):
            if time.time() > deadline:
                return False, None
            
            # Find vertex with min degree
            # To avoid O(n) scan every time, we can keep a list, but for n<2000 it's fine.
            min_deg = min(degrees)
            candidates_u = [i for i in range(n) if degrees[i] == min_deg]
            u = random.choice(candidates_u)
            
            # Find candidate v
            candidates_v = []
            for v in range(n):
                if v == u: continue
                if v in adj[u]: continue
                if degrees[v] >= k: continue
                # Check girth
                # dist(u, v) must be >= g - 1
                # Actually, if dist(u, v) < g - 1, adding edge creates cycle < g.
                # So we need dist(u, v) >= g - 1.
                # But we also need to ensure that adding the edge doesn't create a shorter cycle involving other paths?
                # No, the shortest cycle containing the new edge has length dist(u, v) + 1.
                # If dist(u, v) >= g - 1, then cycle length >= g.
                # Are there other cycles? No, any other cycle not using the new edge already existed.
                # So checking dist(u, v) is sufficient.
                
                # Optimization: if degrees are low, distances are likely large.
                # We can skip BFS if we know the diameter is large.
                # But let's just do BFS for correctness.
                
                # Early exit if we find a valid v
                # We can shuffle candidates to randomize
                pass
                
            # To make it faster, let's collect all valid v's and pick one.
            valid_vs = []
            for v in range(n):
                if v == u or v in adj[u] or degrees[v] >= k:
                    continue
                # Check distance
                # We can limit BFS depth to g-2
                # If dist < g-1, it's invalid.
                # We want dist >= g-1.
                # So if we find a path of length < g-1, it's invalid.
                # BFS until depth g-2. If v is reached, it's invalid.
                
                # Implement BFS with limit
                visited = {u: 0}
                queue = [u]
                found_short = False
                for _ in range(g - 2): # Max depth to check
                    next_queue = []
                    for node in queue:
                        for neighbor in adj[node]:
                            if neighbor == v:
                                found_short = True
                                break
                            if neighbor not in visited:
                                visited[neighbor] = visited[node] + 1
                                next_queue.append(neighbor)
                    if found_short:
                        break
                    queue = next_queue
                    if not queue:
                        break
                
                if not found_short:
                    valid_vs.append(v)
            
            if not valid_vs:
                # Backtrack or restart
                return False, None
            
            v = random.choice(valid_vs)
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        # Check if all degrees are k
        if all(d == k for d in degrees):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return True, edges
        else:
            return False, None

    # Try increasing n
    # Start from n_min
    # We can jump in steps to save time
    current_n = n_min
    step = 10
    while current_n < 2000: # Arbitrary upper bound
        if time.time() > deadline:
            break
            
        success, edges = try_random_greedy(current_n)
        if success:
            return current_n, edges
        
        # Increase n
        current_n += step
        if current_n % 100 == 0:
            # Try a bigger jump if we've been failing
            pass

    # Fallback: LUW graph from seed
    # Re-implement the seed logic to ensure a valid return
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
