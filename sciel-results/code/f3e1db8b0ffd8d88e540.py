import time
import random
import sys
import math

def construct(k, g):
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

    # Moore bound calculation
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))
    
    deadline = time.time() + 110
    
    # Target n=320 for k=4, g=10
    target_n = 320 if k == 4 and g == 10 else moore_bound
    
    # Helper to check girth using non-backtracking walks from every vertex
    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for simple graph
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        
        # Verify degree
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        # Non-backtracking walk check: for each vertex, explore walks of length up to g//2
        # A cycle of length < g exists iff there's a non-backtracking walk that returns to start
        # with length < g, or two non-backtracking walks from same start that meet at distance summing to < g-1
        # Simpler: BFS with parent tracking from each vertex, detect any cycle of length < g
        half_g = g // 2
        for start in range(n):
            # dist[v] = shortest distance from start
            # parent[v] = predecessor on shortest path
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle: dist[u] + dist[v] + 1
                        # But we must ensure it's not just the tree edge
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper to find all short cycles (length < g) efficiently
    def find_short_cycles(n, adj, g):
        """Returns list of cycles (as tuples of vertices) with length < g."""
        cycles = []
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                path1 = []
                                cur = u
                                while cur != start:
                                    path1.append(cur)
                                    cur = parent[cur]
                                path1.append(start)
                                path1.reverse()
                                
                                path2 = []
                                cur = v
                                while cur != start:
                                    path2.append(cur)
                                    cur = parent[cur]
                                path2.append(start)
                                path2.reverse()
                                
                                # Remove the common prefix
                                i = 0
                                while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                                    i += 1
                                cycle = path1[i:] + path2[i-1::-1]
                                if len(cycle) >= 3:
                                    cycles.append(tuple(cycle))
        return cycles

    # Helper to count short cycles (for optimization)
    def count_short_cycles(n, adj, g):
        """Count number of cycles with length < g."""
        cycles = find_short_cycles(n, adj, g)
        return len(cycles)

    # Simulated Annealing for 4-regular graph
    def optimize_girth(n, k, g, time_limit):
        # Generate initial random 4-regular graph
        def random_regular_graph(n, k):
            while True:
                stubs = []
                for i in range(n):
                    for _ in range(k):
                        stubs.append(i)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if (u, v) in seen or (v, u) in seen:
                        valid = False
                        break
                    seen.add((u, v))
                    edges.append((u, v))
                if valid:
                    return edges

        def build_adj(edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def edge_set_from_edges(edges):
            seen = set()
            for u, v in edges:
                if u < v:
                    seen.add((u, v))
                else:
                    seen.add((v, u))
            return seen

        edges = random_regular_graph(n, k)
        adj = build_adj(edges)
        edge_set = edge_set_from_edges(edges)
        
        # Local search: find short cycles and try to break them by edge swaps
        start_time = time.time()
        best_cost = float('inf')
        best_edges = None
        
        while time.time() - start_time < time_limit:
            adj = build_adj(edges)
            cost = count_short_cycles(n, adj, g)
            
            if cost == 0:
                return n, edges
            
            if cost < best_cost:
                best_cost = cost
                best_edges = list(edges)
            
            if best_cost == 0:
                return n, best_edges
            
            # Find a short cycle and try to break it
            cycles = find_short_cycles(n, adj, g)
            if not cycles:
                # Shouldn't happen if cost > 0, but be safe
                break
            
            # Pick a random cycle to break
            cycle = random.choice(cycles)
            cycle_list = list(cycle)
            
            # Try to break the cycle by swapping an edge
            # For each edge in the cycle, try to replace it with a non-cycle edge
            broken = False
            for i in range(len(cycle_list)):
                u = cycle_list[i]
                v = cycle_list[(i + 1) % len(cycle_list)]
                
                # Try to find a replacement edge (u, w) where w is not in the cycle
                # and (u, w) doesn't already exist, and (v, w) doesn't create issues
                candidates = []
                for w in adj[u]:
                    if w == v:
                        continue
                    if (min(u, w), max(u, w)) in edge_set:
                        # Edge (u,w) exists, so we can't add it
                        # But we could remove (u,v) and add (u,w) if (u,w) doesn't exist
                        pass
                    # We want to remove edge (u,v) and add edge (u,w)
                    # where w is not adjacent to u yet
                    if w not in adj[u]:
                        # Check if (u,w) would be valid
                        if (min(u, w), max(u, w)) not in edge_set:
                            # Also need to maintain 4-regularity: remove (u,v) and add (u,w)
                            # But v loses a connection, so we need to add (v, something) too
                            # This is a 2-swap: remove (u,v) and (x,y), add (u,y) and (v,x)
                            # Or simpler: remove (u,v), add (u,w), and find z such that (v,z) can be added
                            # Actually, let's do a proper 2-opt swap
                            pass
                
                # Do a 2-opt swap: remove edges (u,v) and (a,b), add (u,a) and (v,b)
                # where a,b are chosen to break the cycle
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if (min(u, a), max(u, a)) in edge_set:
                        continue
                    if (min(v, a), max(v, a)) in edge_set:
                        continue
                    for b in range(n):
                        if b == u or b == v or b == a:
                            continue
                        if (min(a, b), max(a, b)) in edge_set:
                            continue
                        # Check that removing (u,v) and (a,b) and adding (u,a) and (v,b) is valid
                        # (u,a) must not exist, (v,b) must not exist
                        if (min(v, b), max(v, b)) in edge_set:
                            continue
                        # Also need (a,b) to exist for removal
                        if (min(a, b), max(a, b)) not in edge_set:
                            continue
                        
                        # Perform the swap
                        new_edge_set = set(edge_set)
                        new_edge_set.discard((min(u, v), max(u, v)))
                        new_edge_set.discard((min(a, b), max(a, b)))
                        new_edge_set.add((min(u, a), max(u, a)))
                        new_edge_set.add((min(v, b), max(v, b)))
                        
                        # Check degrees are maintained (they should be, since we remove 2 edges and add 2)
                        # Check no multi-edges
                        valid = True
                        for e in new_edge_set:
                            if e[0] == e[1]:
                                valid = False
                                break
                        if valid:
                            # Build new adj and check cost
                            new_edges = [list(e) for e in new_edge_set]
                            new_adj = build_adj(new_edges)
                            new_cost = count_short_cycles(n, new_adj, g)
                            if new_cost < cost:
                                edges = new_edges
                                edge_set = new_edge_set
                                broken = True
                                break
                    if broken:
                        break
                if broken:
                    break
            
            if not broken:
                # No improvement found, restart
                edges = random_regular_graph(n, k)
                edge_set = edge_set_from_edges(edges)
        
        if best_edges and best_cost == 0:
            return n, best_edges
        return None

    # Try random search for target_n
    if k == 4 and g == 10:
        result = optimize_girth(320, 4, 10, 30)
        if result:
            return result
            
    # Fallback to LUW
    return luw_construct(k, g)
