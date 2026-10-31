import random
import time
import sys

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW construction from seed
    def luw_construct(k, g):
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

    # Check if we are in the specific case where we can do better
    # The prompt highlights k=3, g=16.
    # If k, g are different, we might just use LUW or try the search.
    # For now, let's try the search for any k, g if k is small.
    
    if k == 3 and g == 16:
        # Try to find a smaller graph
        # Known best is 936.
        # Start search from N=936 upwards.
        
        def generate_random_3_regular(n):
            # Configuration model
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                # Normalize edge
                if u > v: u, v = v, u
                if (u, v) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if not valid:
                return None
            return edges

        def check_girth(edges, n, min_girth):
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from each vertex up to depth (min_girth-1)//2
            max_depth = (min_girth - 1) // 2
            for start in range(n):
                visited = {start: 0}
                queue = [(start, 0, -1)] # node, dist, parent
                while queue:
                    node, dist, parent = queue.pop(0)
                    if dist > max_depth:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if neighbor in visited:
                            # Found a cycle
                            cycle_len = dist + 1 + visited[neighbor]
                            if cycle_len < min_girth:
                                return False
                        else:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1, node))
            return True

        # Search
        best_n = None
        best_edges = None
        
        # Try N=936 first, then increase
        # Limit attempts to fit in time
        for n in range(936, 1200, 4):
            if time.time() > deadline - 5:
                break
            attempts = 0
            while attempts < 50 and time.time() < deadline - 5:
                attempts += 1
                edges = generate_random_3_regular(n)
                if edges:
                    if check_girth(edges, n, g):
                        return n, edges
            # If we failed for this n, try larger n
            # Note: Smaller n is better, so we return immediately on success.
            # If we don't find one at 936, we try 940, etc.
            
        # If search failed, return LUW
        return luw_construct(k, g)
        
    else:
        # For other k, g, just use LUW as it's guaranteed valid.
        # We could add similar search logic, but k=3,g=16 is the highlighted case.
        return luw_construct(k, g)
