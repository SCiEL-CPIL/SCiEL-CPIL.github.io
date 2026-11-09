import numpy as np
import random
import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph with girth g using iterative local search.
    Falls back to LUW construction if local search fails.
    """
    start_time = time.time()
    time_limit = 115  # Leave buffer for evaluator

    # Moore bound calculation
    def moore_bound(k, g):
        if k == 2:
            return g
        if g % 2 == 1:
            terms = (g - 1) // 2
            if k == 2:
                return 1 + 2 * terms
            else:
                return 1 + k * (( (k-1)**terms - 1 ) // (k - 2))
        else:
            terms = g // 2 - 1
            if k == 2:
                return 2 * terms
            else:
                return 2 * (( (k-1)**terms - 1 ) // (k - 2))

    n_min = moore_bound(k, g)
    
    # LUW Construction as fallback
    def luw_construct(k, g):
        q = k
        # d is the depth. For LUW, girth is at least d + 5 (for odd girth) or d + 4 (for even)?
        # Actually LUW(q, d) has girth at least 2d+1? No, standard LUW(q, d) has girth >= 2d+1.
        # Let's check: g=9 => 2d+1 >= 9 => 2d >= 8 => d >= 4.
        # n = 2 * q * q^d = 2 * 5 * 5^4 = 6250.
        # If we use d=3, girth >= 7. Not enough for g=9.
        # So d=4 is required for g=9.
        d = (g - 1) // 2
        if d < 1: d = 1
        
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

    # Local Search
    def generate_random_regular_graph(n, k):
        if n * k % 2 != 0 or k >= n:
            return None
        # Configuration model with retries for self-loops/multi-edges
        for _ in range(100):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            edges_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges_set or (v, u) in edges_set:
                    valid = False
                    break
                edges_set.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                # Check degrees
                if all(len(nei) == k for nei in adj):
                    return adj
        return None

    def find_shortest_cycle(adj, g):
        n = len(adj)
        adj_sets = [set(nei) for nei in adj]
        # BFS from each vertex
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            dist = [0] * n
            queue = [start]
            visited[start] = start
            
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        parent[v] = u
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = [u]
                            curr = u
                            while curr != v:
                                curr = parent[curr]
                                cycle.append(curr)
                            cycle.append(v)
                            return cycle
        return None

    def try_local_search(n, k, g, time_limit):
        adj = None
        for _ in range(20):
            adj = generate_random_regular_graph(n, k)
            if adj:
                break
        if not adj:
            return None
            
        adj_sets = [set(nei) for nei in adj]
        
        # We need a fast check. BFS is O(n*k).
        # We will do a limited number of steps.
        steps = 0
        max_steps = 5000
        
        while steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            cycle = find_shortest_cycle(adj, g)
            if not cycle:
                # Success!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
            # Pick a random edge from the cycle
            # cycle is a list of vertices [v0, v1, ..., vL-1]
            # Edges are (v0,v1), (v1,v2), ...
            edge_idx = random.randint(0, len(cycle)-1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx+1) % len(cycle)]
            
            # Try to swap (u,v) with (x,y)
            # Pick random x, y not incident to u or v
            # x must be in adj[u] or adj[v]? No, 2-opt:
            # Remove (u,v) and (x,y). Add (u,x) and (v,y).
            # x and y must be chosen such that (u,x) and (v,y) are valid.
            # x should be a neighbor of some node? 
            # Standard 2-opt for regular graphs:
            # Pick random edge (x,y) in the graph.
            # Ensure x,y not in {u,v} and (u,x),(v,y) not existing.
            
            # To make it faster, we just pick random x, y from the graph's edges.
            # We need a list of edges.
            # We don't have a global edge list readily available in this inner loop efficiently.
            # Let's pick a random vertex x, and a random neighbor y of x.
            
            success_swap = False
            for _ in range(10): # Try a few random swaps
                x = random.randint(0, n-1)
                if x in (u, v): continue
                if not adj[x]: continue
                y = random.choice(adj[x])
                if y in (u, v): continue
                if y == x: continue
                
                # Check if (x,y) is an edge (it is, by construction)
                # Check if (u,x) and (v,y) are valid
                if x in adj[u] or y in adj[v]:
                    continue
                if x == v or y == u: # Avoid multi-edges if u-x and v-y create issues? 
                    # Actually, if x is neighbor of u, we can't add u-x again.
                    pass
                
                # Perform swap
                # Remove (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                # Remove (x,y)
                adj[x].remove(y)
                adj[y].remove(x)
                
                # Add (u,x) and (v,y)
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                success_swap = True
                break
            
            steps += 1
            
        # If we exit without finding a valid graph, return None
        return None

    # Strategy:
    # 1. Estimate n. Start with a multiple of Moore bound.
    # 2. Try local search for a few seconds.
    # 3. If fails, try larger n? Or just fallback to LUW.
    
    # For k=5, g=9, Moore bound is 426. LUW is 6250.
    # Let's try n = 1000, 1200, 1500...
    # But local search is slow.
    # Given the time limit, we might not find 1116.
    # However, even finding a graph with n < 6250 is an improvement.
    
    # Let's try a few n values.
    candidates = [n_min * 2, n_min * 3, n_min * 4, 1000, 1200, 1500]
    # Filter unique and sort
    candidates = sorted(set([c for c in candidates if c >= k+1 and c*k % 2 == 0]))
    
    # If k=5, g=9, n_min=426.
    # candidates: 852, 1278, 1704, 1000, 1200, 1500 -> sorted: 852, 1000, 1200, 1278, 1500, 1704
    
    best_result = None
    for n in candidates:
        if time.time() - start_time > time_limit - 10:
            break
        result = try_local_search(n, k, g, time_limit - 5)
        if result:
            # We found a valid graph!
            # Check if it's better than LUW?
            # LUW n is calculated in fallback.
            # If we found one, it's likely better than LUW if n < LUW_n.
            # Let's just return it.
            return result

    # Fallback to LUW
    return luw_construct(k, g)

