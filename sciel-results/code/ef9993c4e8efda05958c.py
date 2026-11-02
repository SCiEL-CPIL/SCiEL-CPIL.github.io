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

def _greedy_construct(k, g, n, time_limit=10.0):
    start_time = time.time()
    max_restarts = 20 # Adjust based on time
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        adjacency = [set() for _ in range(n)]
        degrees = [0] * n
        
        # Randomize vertex order for processing
        active = list(range(n))
        random.shuffle(active)
        
        success = True
        while active:
            if time.time() - start_time > time_limit:
                success = False
                break
                
            # Pick a vertex to connect
            # Heuristic: Pick the one with the fewest available neighbors?
            # For now, random pick from active
            u_idx = random.randint(0, len(active) - 1)
            u = active[u_idx]
            
            # Find a valid v
            # Shuffle active to avoid bias
            active_copy = active[:]
            random.shuffle(active_copy)
            
            found_v = False
            for v in active_copy:
                if v == u:
                    continue
                if v in adjacency[u]:
                    continue
                
                # Check if adding (u,v) creates a short cycle
                # We need dist(u, v) >= g - 1 in the current graph
                # BFS from u to find v
                if u == v:
                    continue
                    
                visited = {u: 0}
                queue = deque([(u, 0)])
                found_short = False
                while queue:
                    node, dist = queue.popleft()
                    if dist >= g - 1:
                        continue
                    for neighbor in adjacency[node]:
                        if neighbor == v:
                            found_short = True
                            break
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                    if found_short:
                        break
                
                if found_short:
                    continue
                
                # Add edge
                adjacency[u].add(v)
                adjacency[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                found_v = True
                break
            
            if not found_v:
                success = False
                break
                
            # Update active list
            # Remove u if degree == k
            # Remove v if degree == k
            # This is O(n).
            new_active = []
            for node in active:
                if degrees[node] < k:
                    new_active.append(node)
            active = new_active
            
        if success:
            # Verify
            for i in range(n):
                if len(adjacency[i]) != k:
                    success = False
                    break
            
            if success:
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    return None

def construct(k, g):
    total_start = time.time()
    hard_deadline = 140.0
    
    if k == 5 and g == 9:
        # Try greedy with smaller n
        # Known optimum is 1116. LUW is 1250.
        # Try 1116, 1150, 1200
        target_ns = [1116, 1150, 1200]
        
        for n in target_ns:
            elapsed = time.time() - total_start
            if elapsed > hard_deadline - 10:
                break
            time_per_attempt = min(30.0, hard_deadline - elapsed - 5)
            if time_per_attempt < 2.0:
                break
            res = _greedy_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
    # Fallback to LUW
    return _luw_construct(k, g)
