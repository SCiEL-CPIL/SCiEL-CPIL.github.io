import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
        
    best_n = None
    best_edges = None
    
    # Helper to check girth >= g, returns True/False
    def has_girth_at_least_g(adj, n, g):
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1))
            
            while queue:
                new_queue = []
                for prev, curr, depth in queue:
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            if depth + 1 < g:
                                return False
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_queue.append((curr, nxt, depth + 1))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
        return True

    # Returns (min_cycle_length, list_of_edges_in_shortest_cycle)
    def find_shortest_cycle(adj, n, max_girth):
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        min_cycle = max_girth + 1
        best_cycle_edges = None
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1, [(start, nb)]))
            
            while queue:
                new_queue = []
                for prev, curr, depth, path in queue:
                    if depth >= max_girth - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            cycle_len = depth + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                                # Reconstruct cycle edges: path + (curr, start)
                                cycle_edges = list(path) + [(curr, start)]
                                best_cycle_edges = cycle_edges
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_path = path + [(curr, nxt)]
                                new_queue.append((curr, nxt, depth + 1, new_path))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
            if min_cycle == 3:
                break
        return min_cycle, best_cycle_edges

    def random_regular_graph(n, k, rng):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        seen = [set() for _ in range(n)]
        
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if v in seen[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            seen[u].add(v)
            seen[v].add(u)
            
        return adj

    def get_all_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    def try_swap(adj, n, k, rng, prefer_edges=None):
        edges = get_all_edges(adj, n)
        if len(edges) < 2:
            return False
            
        if prefer_edges and len(prefer_edges) >= 2:
            # Try to pick one edge from prefer_edges and one random edge
            e1 = rng.choice(prefer_edges)
            e2 = rng.choice(edges)
            if e1 == e2:
                e2 = rng.choice(edges)
                if e1 == e2:
                    return False
        else:
            e1_idx = rng.randint(0, len(edges) - 1)
            e2_idx = rng.randint(0, len(edges) - 2)
            if e2_idx >= e1_idx:
                e2_idx += 1
            e1 = edges[e1_idx]
            e2 = edges[e2_idx]
            
        u1, v1 = e1
        u2, v2 = e2
        
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return False
            
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        if u2 not in adj[u1] and v2 not in adj[v1]:
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        
        if v2 not in adj[u1] and u2 not in adj[v1]:
            adj[u1].append(v2)
            adj[v2].append(u1)
            adj[v1].append(u2)
            adj[u2].append(v1)
            return True
            
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        
        return False

    # Main search loop
    n_start = moore
    if (n_start * k) % 2 != 0:
        n_start += 1
        
    max_n_to_try = 500
    
    for n in range(n_start, max_n_to_try + 1, 2):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline - 5:
            adj = random_regular_graph(n, k, rng)
            if adj is None:
                attempts += 1
                continue
                
            if has_girth_at_least_g(adj, n, g):
                edges = get_all_edges(adj, n)
                return n, edges
            
            # Local search: repeatedly find shortest cycle, try to break it
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                
                min_cycle, cycle_edges = find_shortest_cycle(adj, n, g)
                
                if min_cycle >= g:
                    edges = get_all_edges(adj, n)
                    return n, edges
                
                # Try to break the shortest cycle by swapping one of its edges
                # with a random edge
                if cycle_edges:
                    # Normalize cycle edges to canonical form
                    canon_cycle = []
                    for a, b in cycle_edges:
                        canon_cycle.append((min(a,b), max(a,b)))
                    
                    # Try multiple times to break this cycle
                    for _attempt in range(10):
                        if time.time() > deadline - 5:
                            break
                        if try_swap(adj, n, k, rng, prefer_edges=canon_cycle):
                            break
                else:
                    try_swap(adj, n, k, rng)
                
                # Check girth periodically
                if _ % 20 == 0:
                    if has_girth_at_least_g(adj, n, g):
                        edges = get_all_edges(adj, n)
                        return n, edges
            
            attempts += 1
            
    # Fallback: LUW construction
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
        if time.time() > deadline - 2:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
