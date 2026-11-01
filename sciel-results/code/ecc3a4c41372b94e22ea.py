import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Safety margin (evaluator kills at 150s)

    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Heuristic Search for k=3, g=16
    if k == 3 and g == 16:
        best_n, best_edges = luw_construct(k, g)

        def build_adj(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def find_short_cycle(adj, min_girth):
            """Find a cycle shorter than min_girth, return (length, cycle) or None."""
            n = len(adj)
            max_depth = (min_girth - 1) // 2
            for start in range(n):
                visited = [-1] * n
                visited[start] = 0
                parent = [-1] * n
                queue = deque([(start, 0, -1)])
                while queue:
                    node, depth, par = queue.popleft()
                    if depth > max_depth:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == par:
                            continue
                        if visited[neighbor] == -1:
                            visited[neighbor] = depth + 1
                            parent[neighbor] = node
                            queue.append((neighbor, depth + 1, node))
                        else:
                            cycle_len = depth + visited[neighbor] + 1
                            if cycle_len < min_girth:
                                # Reconstruct cycle
                                cycle = [neighbor]
                                cur = node
                                while cur != neighbor:
                                    cur = parent[cur]
                                    if cur == -1:
                                        break
                                    cycle.append(cur)
                                cycle.append(neighbor)
                                return cycle_len, cycle
            return None

        def local_search(adj, min_girth, max_iter=5000):
            """Try to fix short cycles via edge swaps."""
            n = len(adj)
            for iteration in range(max_iter):
                if time.time() > deadline - 10:
                    break
                sc = find_short_cycle(adj, min_girth)
                if sc is None:
                    return True, adj
                cycle_len, cycle = sc
                # Try to break the cycle by swapping an edge
                for i in range(len(cycle) - 1):
                    u = cycle[i]
                    v = cycle[i + 1]
                    # Find alternative neighbors to swap
                    for u_alt in adj[u]:
                        if u_alt == v:
                            continue
                        for v_alt in adj[v]:
                            if v_alt == u:
                                continue
                            # Check if swap is valid (no multi-edge, no loop)
                            if u_alt == v or v_alt == u:
                                continue
                            if u_alt in adj[v] or v_alt in adj[u]:
                                continue
                            # Perform swap: remove (u,v), add (u,v_alt) and (v,u_alt)
                            # Actually, we need to remove (u,v) and add two new edges
                            # But that changes degree. Let's do a 2-opt style swap.
                            # Remove (u,v) and (u_alt, v_alt) if they exist, add (u,v_alt) and (v,u_alt)
                            # Simpler: just try to rewire one endpoint
                            pass
                
                # Simpler approach: random edge swap
                # Pick two random non-adjacent vertices and swap one of their edges
                if random.random() < 0.3:
                    # Find a random edge to potentially rewire
                    u = random.randint(0, n - 1)
                    v = random.choice(adj[u])
                    # Remove edge (u, v)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    # Find new neighbors
                    candidates = [x for x in range(n) if x not in adj[u] and x not in adj[v] and x != u and x != v]
                    if candidates:
                        new_v = random.choice(candidates)
                        adj[u].append(new_v)
                        adj[new_v].append(u)
                    else:
                        # Undo
                        adj[u].append(v)
                        adj[v].append(u)
            
            return False, adj

        def random_3reg(n):
            """Generate a random 3-regular graph on n vertices."""
            if n % 2 != 0:
                return None
            # Configuration model
            half_edges = [i for i in range(n) for _ in range(3)]
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    return None  # Loop, reject
                if v in adj[u]:
                    return None  # Multi-edge, reject
                adj[u].append(v)
                adj[v].append(u)
            return adj

        # Strategy: Try random 3-regular graphs with local search
        # Start from Moore bound 512, try n = 512, 514, 516, ...
        # But n must be even for 3-regular.
        
        search_deadline = start_time + 95.0  # Leave time for verification
        
        best_found_n = best_n
        best_found_edges = best_edges
        
        for n in range(512, 700, 2):  # Even n only
            if time.time() > search_deadline:
                break
            if n >= best_found_n:
                break
            
            found = False
            max_attempts = 200  # Try up to 200 random graphs
            
            for attempt in range(max_attempts):
                if time.time() > search_deadline:
                    break
                
                adj = random_3reg(n)
                if adj is None:
                    continue
                
                # Check if already valid
                sc = find_short_cycle(adj, 16)
                if sc is None:
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    if n < best_found_n:
                        best_found_n = n
                        best_found_edges = edges
                    found = True
                    break
                
                # Try local search to fix short cycles
                valid, adj_fixed = local_search(adj, 16, max_iter=2000)
                if valid:
                    edges = []
                    for i in range(n):
                        for j in adj_fixed[i]:
                            if i < j:
                                edges.append((i, j))
                    if n < best_found_n:
                        best_found_n = n
                        best_found_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if best_found_n < best_n:
            return best_found_n, best_found_edges

        return best_n, best_edges

    return luw_construct(k, g)
# EVOLVE-BLOCK-END
