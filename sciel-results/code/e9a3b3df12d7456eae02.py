import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Seed construction for fallback
    def seed_construct(k, g):
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

    best_n, best_edges = seed_construct(k, g)
    
    # Helper to check girth
    def check_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles < target_girth
        # BFS from each node
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= target_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    def is_valid(n, edges):
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return False
        # Check simple
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return check_girth(n, edges, g)

    # Random 3-regular graph generation
    def gen_random_3reg(n):
        if n % 2 != 0:
            return None
        # Configuration model
        half_edges = n * k // 2
        # List of stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                edges.append((u, v))
            else:
                # Loop, retry or fix
                return None
        # Check for multi-edges
        seen = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in seen:
                return None
            seen.add(key)
        return edges

    # Local search
    def local_search(n, edges, max_swaps):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find short cycles
        short_cycles = []
        # Simplified: just check girth
        if not check_girth(n, edges, g):
            # Try swaps
            for _ in range(max_swaps):
                if time.time() > deadline:
                    break
                # Pick random edge
                e1_idx = random.randint(0, len(edges) - 1)
                u1, v1 = edges[e1_idx]
                # Pick random other edge
                e2_idx = random.randint(0, len(edges) - 1)
                if e2_idx == e1_idx:
                    continue
                u2, v2 = edges[e2_idx]
                
                # Try swap (u1, v1) and (u2, v2) -> (u1, u2) and (v1, v2)
                # Or (u1, v2) and (v1, u2)
                
                # Check if new edges exist
                new_e1 = (u1, u2)
                new_e2 = (v1, v2)
                
                # Ensure no loops
                if u1 == u2 or v1 == v2:
                    continue
                    
                # Ensure no multi-edges
                # Remove old, add new, check
                # This is slow. 
                # Optimization: Only accept if it reduces the number of short cycles?
                # For now, just try and check girth.
                
                # Construct new edge list
                new_edges = edges[:]
                new_edges[e1_idx] = new_e1
                new_edges[e2_idx] = new_e2
                
                if is_valid(n, new_edges):
                    return new_edges
        return None

    # Main loop
    # Try to find smaller N
    for N in range(935, 700, -5):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            edges = gen_random_3reg(N)
            if edges is None:
                continue
            # Try to improve girth
            improved_edges = local_search(N, edges, 1000)
            if improved_edges is not None:
                best_n, best_edges = N, improved_edges
                found = True
                break
        if found:
            # Try to go smaller?
            # Continue loop to try smaller N
            pass
        else:
            # If we failed for this N, maybe try smaller?
            # But if we can't fix girth for N, we likely can't for N-5.
            # However, smaller N is harder to satisfy girth.
            # So if we fail at N, we should stop?
            # No, random graphs of smaller N have lower girth on average.
            # So it's harder.
            # If we found a valid graph for N, we can try N-5.
            # If we didn't, we should probably stop or try a different strategy.
            # Let's stop if we fail to find any valid graph for a few consecutive N.
            # For simplicity, just break if we fail.
            break

    return best_n, best_edges
