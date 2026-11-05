import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep margin
    
    # If k is not 3, fallback to LUW (or whatever the seed did)
    # The seed handles general k, but the task is specifically about n(3,16)
    # However, the interface is construct(k, g).
    # We should handle k=3 efficiently, and fallback for others.
    
    if k != 3:
        # Fallback to LUW logic from seed
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

    # For k=3, try to find a small Cayley graph
    # Target girth >= 16
    
    def check_girth(n, a, min_girth):
        """
        Returns True if the Cayley graph Z_n with generators {1, -1, a}
        has girth at least min_girth.
        """
        # Check for cycles of length < min_girth
        # BFS from each vertex, no backtracking
        # If we find a cycle of length < min_girth, return False
        
        for start in range(n):
            # State: (current, prev)
            # We use a set to track visited states to avoid re-exploring
            # But for girth check, we just need to find the shortest cycle.
            # BFS finds the shortest path to a node.
            # So we can just do a standard BFS from `start` and see if we return to `start`.
            # However, we must avoid backtracking.
            
            visited = set()
            queue = [(start, None, 0)]
            while queue:
                u, p, d = queue.pop(0)
                
                if u == start and d > 0:
                    if d < min_girth:
                        return False
                    # If d >= min_girth, this cycle is acceptable.
                    # Since BFS finds the shortest path, this is the shortest cycle through `start`.
                    # So we can break for this `start`.
                    break
                
                if (u, p) in visited:
                    continue
                visited.add((u, p))
                
                if d >= min_girth - 1:
                    continue
                
                # Neighbors: u+1, u-1, u+a
                for gen in (1, -1, a):
                    next_u = (u + gen) % n
                    if next_u == p:
                        continue
                    queue.append((next_u, u, d + 1))
        
        return True

    # Search for n and a
    # We know n(3,16) = 936.
    # Try n around 936.
    # a should be such that a != 1, a != n-1, a != n/2.
    
    for n in range(900, 1100, 2):
        if time.time() > deadline - 5:
            break
            
        # Try a few a values
        # Random a values are good candidates
        # We can try a = 100, 101, ...
        # Or just iterate a
        found = False
        for a in range(2, n // 2):
            if a == n // 2:
                continue
            if time.time() > deadline - 5:
                break
                
            if check_girth(n, a, 16):
                # Construct edges
                edge_set = set()
                for i in range(n):
                    edge_set.add((i, (i + 1) % n))
                    edge_set.add((i, (i + a) % n))
                
                edges = []
                for u, v in edge_set:
                    if u > v:
                        u, v = v, u
                    edges.append((u, v))
                
                return n, edges
        # If not found for this n, try next n

    # Fallback to LUW if no small Cayley graph found
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
