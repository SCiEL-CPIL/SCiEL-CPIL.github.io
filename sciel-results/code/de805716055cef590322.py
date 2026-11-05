import time
import random
import itertools
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k != 3:
        # Fallback: LUW construction for general k
        return luw_construct(k, g)

    # For k=3, g=16: use randomized edge-switching to build a 3-regular graph
    # with girth >= 16, minimizing vertex count.
    # Strategy: Start from a random 3-regular graph, then iteratively perform
    # edge switches that reduce the number of short cycles.
    # We try to find a graph with as few vertices as possible.

    # Moore bound for (3,16): 2*(1+3+9+27+81+243) = 2*364 = 728? 
    # Actually for k=3, g=16 (even), Moore bound = 2*(k^(g/2-1) + k^(g/2-2) + ... + k) 
    # Wait, let me compute properly.
    # For even girth g, Moore bound M(k,g) = 2 * sum_{i=0}^{g/2-1} (k-1)^i * k^0 ... 
    # Actually: M = 2 * (k-1) * ((k-1)^(g/2) - 1) / (k-2) for k>2
    # For k=3, g=16: M = 2*2*((2)^8 - 1)/(1) = 4*(255) = 1020? That seems high.
    # Let me use the standard formula: for girth g, degree k:
    # If g is even: M >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * ((k-1)^(g/2) - 1)/(k-2)
    # For k=3, g=16: 2 * (2^8 - 1)/(2-1) = 2 * 255 = 510. So lower bound is 510.
    # But the task says best lower bound is 512. Close enough.
    # The smallest known is 936.

    # We'll try to build graphs starting from smaller n and increase.
    # Start from a reasonable size and use edge switching.

    def count_short_cycles(adj, n, max_len):
        """Count non-backtracking walks of length up to max_len from all vertices."""
        # This is expensive; we'll use a simpler heuristic: count cycles of length < g
        # by doing BFS from each vertex and counting cycles found.
        # For efficiency, we only count cycles of length up to max_len.
        total = 0
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g and cycle_len <= max_len:
                            total += 1
        return total

    def girth_check(adj, n, g):
        """Check if all cycles have length >= g."""
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] >= g:
                    continue
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return False
        return True

    def random_3regular(n):
        """Generate a random 3-regular graph on n vertices. n must be even."""
        if n % 2 != 0 or n < 4:
            return None
        # Configuration model
        half_edges = list(range(n)) * 3
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        edges = set()
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                continue
            key = (min(u,v), max(u,v))
            if key in edges:
                continue
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        # Check if all vertices have degree 3
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj, sorted(edges)

    def edge_switch(adj, edges, u1, v1, u2, v2):
        """Remove edges (u1,v1) and (u2,v2), add edges (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)."""
        # Try both possible reconnections
        for (a, b, c, d) in [(u1, u2, v1, v2), (u1, v2, v1, u2)]:
            if a == b or c == d:
                continue
            key1 = (min(a,b), max(a,b))
            key2 = (min(c,d), max(c,d))
            if key1 in edges or key2 in edges:
                continue
            # Perform switch
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)
            return True, (a,b), (c,d)
        return False, None, None

    best_n = None
    best_edges = None

    # Try different starting sizes
    for start_n in [512, 520, 528, 536, 544, 552, 560, 568, 576, 584, 592, 600, 640, 680, 720, 760, 800, 840, 880, 920, 936]:
        if time.time() > deadline - 15:
            break

        # Try multiple random initializations
        found = False
        for trial in range(20):
            if time.time() > deadline - 10:
                break

            result = random_3regular(start_n)
            if result is None:
                continue
            adj, edges_list = result
            edges_set = set((min(i,j), max(i,j)) for i,j in edges_list)

            # Check initial girth
            if girth_check(adj, start_n, g):
                if best_n is None or start_n < best_n:
                    best_n = start_n
                    best_edges = [(min(i,j), max(i,j)) for i,j in edges_set]
                found = True
                break

            # Try to improve via edge switching
            # Count initial short cycles
            initial_cycles = count_short_cycles(adj, start_n, g-1)

            for step in range(500):
                if time.time() > deadline - 10:
                    break

                # Pick two random edges
                edge_list = list(edges_set)
                if len(edge_list) < 2:
                    break
                e1, e2 = random.sample(edge_list, 2)
                u1, v1 = e1
                u2, v2 = e2

                # Ensure they don't share vertices
                if len({u1, v1, u2, v2}) < 4:
                    continue

                # Try the switch
                success, new_e1, new_e2 = edge_switch(adj, edges_set, u1, v1, u2, v2)
                if not success:
                    continue

                # Check if girth improved
                new_cycles = count_short_cycles(adj, start_n, g-1)
                if new_cycles < initial_cycles:
                    initial_cycles = new_cycles
                    edges_set.add(new_e1)
                    edges_set.add(new_e2)
                    # Check if we've achieved girth >= g
                    if girth_check(adj, start_n, g):
                        if best_n is None or start_n < best_n:
                            best_n = start_n
                            best_edges = [(min(i,j), max(i,j)) for i,j in edges_set]
                        found = True
                        break
                else:
                    # Revert
                    a, b = new_e1
                    c, d = new_e2
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)

            if found:
                break

        if best_n is not None and best_n <= start_n:
            # We found a valid graph at this size or smaller
            break

    if best_n is not None:
        return best_n, best_edges

    # Fallback: LUW construction
    return luw_construct(3, g)

def luw_construct(k, g):
    q = k
    d = max(3, (g - 2) // 2)
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
