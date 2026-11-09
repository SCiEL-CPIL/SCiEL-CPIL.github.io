import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
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

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # Specialized search for 3-regular, girth 16
    for n_target in [700, 800, 900, 1000]:
        if time.time() > deadline - 30:
            break
        
        # Generate random 3-regular graph
        adj = None
        for _ in range(10): # Retry generation if invalid
            half_edges = [i for i in range(n_target) for _ in range(3)]
            random.shuffle(half_edges)
            adj = {i: [] for i in range(n_target)}
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                break
        
        if not adj:
            continue

        def find_shortest_cycle(adj, n):
            # BFS from each node
            min_len = float('inf')
            min_cycle = None
            for start in range(n):
                dist = {start: 0}
                parent = {start: -1}
                queue = [start]
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if v == parent.get(u, -1):
                            continue
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Cycle found
                            # Reconstruct path
                            # This is expensive. Just return length for now.
                            # To get the cycle, we need to trace back.
                            # For now, just return length.
                            # If we need the cycle edges, we trace.
                            pass
            # This implementation doesn't return the cycle edges.
            # We need the cycle edges to break them.
            # Let's modify to return the cycle.
            return min_len, min_cycle

        # Better: Find one short cycle, break it, repeat.
        # We don't need the *shortest* necessarily, just *a* short cycle.
        # But breaking the shortest is best.
        
        for step in range(2000):
            if time.time() > deadline - 10:
                break
            
            # Find a short cycle (length < g)
            # Use BFS from a random node to find a cycle quickly?
            # Or just check girth every 100 steps.
            # Let's find a cycle of length < g.
            
            # Simple check: Pick a random node, BFS up to depth g/2.
            # If we find a back edge, we have a cycle.
            # Let's just try random swaps and check girth every 50 steps.
            
            # Pick random edge
            u = random.randint(0, n_target - 1)
            v = random.choice(adj[u])
            
            # Pick another random edge
            x = random.randint(0, n_target - 1)
            y = random.choice(adj[x])
            
            if (u, v) == (x, y) or (u, v) == (y, x):
                continue
            if u == x or u == y or v == x or v == y:
                continue
            if y in adj[u] or x in adj[v]:
                continue
                
            # Swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            if step % 50 == 0:
                # Check girth
                # BFS from each node, depth g/2 - 1
                # If no cycle found, girth > g/2 * 2 = g
                # Actually, if no cycle of length < g, we are good.
                # BFS depth d finds cycles of length <= 2d + 1?
                # No, BFS from a node finds shortest cycles passing through that node.
                # To find global girth, we need to check all.
                # For speed, we can sample 100 random nodes.
                girth_ok = True
                for start in random.sample(range(n_target), min(100, n_target)):
                    dist = {start: 0}
                    parent = {start: -1}
                    queue = [start]
                    found = False
                    while queue:
                        u = queue.pop(0)
                        if dist[u] >= g // 2:
                            continue
                        for v in adj[u]:
                            if v == parent.get(u, -1):
                                continue
                            if v not in dist:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                # Cycle
                                # Check length
                                # If dist[u] + dist[v] + 1 < g, then girth < g
                                # This is an approximation.
                                # For exact, we need full BFS.
                                # Given time, we assume if we don't find short cycles in sample, it's good.
                                # But we need to be sure.
                                # Let's do full girth check if sample looks good.
                                pass
                    # If we didn't find any short cycle in this BFS, it's promising.
                
                # Full check (expensive)
                # Only do full check if sample suggests girth >= g
                # For now, let's just do a full check every 200 steps.
                if step % 200 == 0:
                    min_girth = float('inf')
                    for start in range(n_target):
                        dist = {start: 0}
                        parent = {start: -1}
                        queue = [start]
                        while queue:
                            u = queue.pop(0)
                            if dist[u] >= g // 2:
                                continue
                            for v in adj[u]:
                                if v == parent.get(u, -1):
                                    continue
                                if v not in dist:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    queue.append(v)
                                else:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < min_girth:
                                        min_girth = cycle_len
                    if min_girth >= g:
                        edges = []
                        for i in range(n_target):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return n_target, edges

    return luw_construct(k, g)
