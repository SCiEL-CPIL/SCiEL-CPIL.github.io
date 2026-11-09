import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if not (k == 5 and g == 9):
        return _seed_construct(k, g)

    # Try to find a graph with n < 1250
    # We will try local search on n=1116, 1120, 1150
    # If we find a valid graph, return it.
    # Otherwise, return seed.
    
    n_candidates = [1116, 1120, 1150]
    
    for n in n_candidates:
        if time.time() > deadline - 10:
            break
        result = _attempt_local_search(n, k, g, deadline - 5)
        if result is not None:
            return result

    return _seed_construct(k, g)

def _attempt_local_search(n, k, g, deadline):
    # Generate random k-regular graph
    # Try to increase girth via 2-switches targeting short cycles
    
    for restart in range(3):
        if time.time() > deadline:
            return None
        
        adj, is_simple = generate_random_regular_graph(n, k)
        if not is_simple:
            continue
            
        # Local search
        adj = _local_search_girth(adj, n, k, g, deadline)
        
        # Check final girth
        girth = compute_girth(adj, n)
        if girth >= g:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges

    return None

def _local_search_girth(adj, n, k, g, deadline):
    # Loop until time or girth >= g
    # We use a "cycle breaking" strategy with targeted swap selection.
    
    # Precompute adjacency sets for fast lookup
    adj_sets = [set(s) for s in adj]
    
    while time.time() < deadline:
        # Find a cycle of length < g
        cycle = find_short_cycle(adj, n, g)
        if cycle is None:
            return adj # Girth >= g
        
        # Pick a random edge from the cycle to break
        L = len(cycle)
        idx = random.randint(0, L - 1)
        u = cycle[idx]
        v = cycle[(idx + 1) % L]
        
        # Ensure u < v for consistency
        if u > v:
            u, v = v, u
        
        # Find candidate edges (w, x) to swap with
        # Prefer edges that are far from u and v to avoid creating short cycles
        # Collect all edges not involving u or v
        candidate_edges = []
        for i in range(n):
            if i == u or i == v:
                continue
            for j in adj[i]:
                if j > i and j != u and j != v:
                    candidate_edges.append((i, j))
        
        if not candidate_edges:
            break
        
        # Try swaps, preferring candidates farther from the cycle
        found_swap = False
        for _ in range(50):
            w, x = random.choice(candidate_edges)
            
            # Check validity of swap (u,v), (w,x) -> (u,w), (v,x)
            if w in adj_sets[u] or x in adj_sets[v]:
                continue
            
            # Check no triangles created
            common_uw = adj_sets[u] & adj_sets[w]
            common_vx = adj_sets[v] & adj_sets[x]
            
            if common_uw or common_vx:
                continue

            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj_sets[u].discard(v)
            adj_sets[v].discard(u)
            adj_sets[w].discard(x)
            adj_sets[x].discard(w)
            
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            adj_sets[u].add(w)
            adj_sets[w].add(u)
            adj_sets[v].add(x)
            adj_sets[x].add(v)
            
            found_swap = True
            break
            
        if not found_swap:
            break

    return adj

def find_short_cycle(adj, n, g):
    # BFS from each node to find shortest cycle
    # Return the cycle path if length < g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        path_u = [u]
                        while path_u[-1] != start:
                            path_u.append(parent[path_u[-1]])
                        path_u.reverse()
                        
                        path_v = [v]
                        while path_v[-1] != start:
                            path_v.append(parent[path_v[-1]])
                        path_v.reverse()
                        
                        # Cycle is path_u + path_v (excluding start duplicate?)
                        # path_u: start ... u
                        # path_v: start ... v
                        # Cycle: u ... start ... v ... u ?
                        # So list is path_u (reversed to u->start) + path_v (start->v)
                        # Wait, path_u is start->...->u. Reversed is u->...->start.
                        # path_v is start->...->v.
                        # So cycle is path_u_reversed + path_v[1:]
                        # Let's verify.
                        # u -> ... -> start -> ... -> v -> u (edge v-u exists? No, v and u are neighbors in BFS tree? No, they are connected by an edge in the graph, but not in the tree necessarily. The edge (u,v) closes the cycle.)
                        # So the cycle is: u, ..., start, ..., v, u.
                        cycle = list(reversed(path_u)) + path_v[1:]
                        return cycle
    return None

def generate_random_regular_graph(n, k):
    if (n * k) % 2 != 0:
        return None, False
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None, False
        if v in adj[u]:
            return None, False
        adj[u].add(v)
        adj[v].add(u)
        edges.add((min(u, v), max(u, v)))
    adj_list = [list(s) for s in adj]
    return adj_list, True

def compute_girth(adj_list, n):
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth == 3:
                        return 3
    return min_girth if min_girth != float('inf') else n

def _seed_construct(k, g):
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
