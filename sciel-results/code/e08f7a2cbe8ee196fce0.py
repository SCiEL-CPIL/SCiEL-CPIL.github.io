import time
import random
import sys
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_graph():
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

    # Helper: Generate random k-regular graph
    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        half_edges = []
        for i in range(n):
            half_edges.extend([i] * k)
        
        while True:
            random.shuffle(half_edges)
            edges = []
            valid = True
            seen = set()
            for i in range(0, n * k, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if valid:
                return edges

    # Helper: Check girth using BFS from every vertex
    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
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
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper: Build adjacency list and set for fast edge lookup
    def build_adjacency(n, edges):
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            if u > v:
                u, v = v, u
            edge_set.add((u, v))
        return adj, edge_set

    # Helper: Find all cycles of length L in the graph
    def find_cycles_of_length(n, adj, L):
        cycles = []
        visited = set()
        
        for start in range(n):
            # Only start from the smallest vertex in the cycle to avoid duplicates
            for first in adj[start]:
                if first < start:
                    continue
                # DFS to find cycles of length L
                def dfs(current, path, parent):
                    if len(path) == L:
                        if current == start:
                            # Found a cycle
                            cycle = tuple(sorted(path[:-1]))
                            if cycle not in visited:
                                visited.add(cycle)
                                cycles.append(list(path[:-1]))
                        return
                    if len(path) >= L:
                        return
                    for neighbor in adj[current]:
                        if neighbor == parent:
                            continue
                        if neighbor == start and len(path) < L:
                            continue
                        if neighbor in path:
                            continue
                        dfs(neighbor, path + [neighbor], current)
                
                dfs(first, [start, first], start)
        
        return cycles

    # Helper: Try to extend an 8-cycle to a 9-cycle via edge switching
    def try_extend_cycle(n, adj, edge_set, cycle, k):
        # cycle is a list of 8 vertices forming an 8-cycle
        # We want to find two edges in the cycle that can be swapped to break the 8-cycle
        # while maintaining k-regularity
        
        # For each pair of non-adjacent edges in the cycle, try swapping
        # Edge (u, v) and edge (w, x) can be swapped to (u, x) and (w, v) or (u, w) and (v, x)
        # if the new edges don't already exist and don't create multi-edges
        
        L = len(cycle)
        for i in range(L):
            u1 = cycle[i]
            v1 = cycle[(i + 1) % L]
            for j in range(i + 2, L):
                if i == 0 and j == L - 1:
                    continue  # These are adjacent in the cycle
                u2 = cycle[j]
                v2 = cycle[(j + 1) % L]
                
                # Try swapping: remove (u1, v1) and (u2, v2), add (u1, u2) and (v1, v2)
                if (min(u1, u2), max(u1, u2)) in edge_set or (min(v1, v2), max(v1, v2)) in edge_set:
                    continue
                if u1 == u2 or v1 == v2:
                    continue
                
                # Perform the swap
                old_edges = [(min(u1, v1), max(u1, v1)), (min(u2, v2), max(u2, v2))]
                new_edges = [(min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))]
                
                # Check that this doesn't create a shorter cycle
                # This is expensive, so we'll just try it and check
                for u in [u1, v1, u2, v2]:
                    adj[u].remove(v1 if u == u1 else u1 if u == v1 else v2 if u == u2 else u2)
                adj[u1].append(u2)
                adj[v1].append(v2)
                adj[u2].append(u1)
                adj[v2].append(v1)
                
                edge_set.remove(old_edges[0])
                edge_set.remove(old_edges[1])
                edge_set.add(new_edges[0])
                edge_set.add(new_edges[1])
                
                # Check if the girth is still >= g
                if check_girth(n, list(edge_set), g):
                    return True, new_edges, old_edges
                
                # Undo the swap
                adj[u1].remove(u2)
                adj[v1].remove(v2)
                adj[u2].remove(u1)
                adj[v2].remove(v1)
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                edge_set.remove(new_edges[0])
                edge_set.remove(new_edges[1])
                edge_set.add(old_edges[0])
                edge_set.add(old_edges[1])
        
        return False, None, None

    # Main Logic
    # If k=5 and g=9, try to beat the baseline (1250)
    if k == 5 and g == 9:
        # Try N values. 1116 is known to exist.
        # Start with 1116, then 1200, then 1000.
        target_ns = [1116, 1150, 1200, 1000, 900]
        
        for n in target_ns:
            if time.time() > deadline - 10:
                break
                
            # Try a few random instances
            max_attempts = 50
            for _ in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                edges = random_regular_graph(n, k)
                if edges is not None:
                    if check_girth(n, edges, g):
                        return n, edges
                        
    # Fallback to LUW
    return luw_graph()
