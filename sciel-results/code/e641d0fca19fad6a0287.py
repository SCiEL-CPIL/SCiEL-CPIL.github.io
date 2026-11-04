"""Voltage-graph lift over Z_m for a 3-regular graph with large girth."""
import time
import random

def construct(k, g):
    # k=3, g=16
    # Build a 3-regular graph with girth >= g using voltage graph / circulant approach.
    # Strategy: use a known small 3-regular graph with good girth properties as a
    # "base" and lift it. For k=3, the Petersen graph (girth 5) is a natural base.
    # We use a voltage assignment over Z_m to create a 3m-vertex graph.
    #
    # Actually, for the cage problem n(3,16), the best known is 936 vertices.
    # The Moore bound for (3,16) is 2*(3^(g/2-1) - 1) + 2*3^(g/2-1) ... let me compute.
    # For girth g (even), Moore bound: 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*(255) = 510.
    # So we need at least 510 vertices, best known is 936.
    #
    # Approach: Use a random 3-regular graph and try to maximize girth via edge swaps.
    # But that's hard. Instead, use the known LUW construction but with smaller parameters.
    #
    # Actually, let me use a simpler approach: construct a circulant-like graph.
    # A 3-regular circulant graph on n vertices has edges (i, i+a), (i, i-b), (i, i+c) mod n.
    # For girth 16, we need n to be large enough.
    #
    # Let me use the standard approach: for 3-regular graphs with large girth,
    # the best constructions use algebraic methods. Let me try a product/circulant approach.
    #
    # For a 3-regular graph on n vertices with girth >= 16:
    # The number of non-backtracking walks of length L from a vertex is at most (k-1)^L = 2^L.
    # For girth g, no closed non-backtracking walks of length < g.
    #
    # Let me use a random regular graph with local search to maximize girth.
    # Start with a random 3-regular graph on n vertices, then do edge swaps to increase girth.
    
    deadline = time.time() + 100  # safety margin before 120s limit
    
    def is_valid_graph(n, edges):
        """Check if graph is simple, 3-regular, and has girth >= g."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            if v in adj[u]:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check girth using BFS from each vertex (or sample)
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            parent = [-1] * n
            from collections import deque
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True
    
    def girth_of_graph(n, adj):
        """Compute girth of a graph using BFS from each vertex."""
        from collections import deque
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            parent = [-1] * n
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= min_girth // 2:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if dist[u] >= min_girth // 2:
                            break
        return min_girth if min_girth != float('inf') else 999
    
    def random_3_regular(n):
        """Generate a random 3-regular graph on n vertices (n must be even)."""
        if n % 2 != 0:
            n += 1
        # Use the configuration model
        half_edges = list(range(n))
        for _ in range(3):
            half_edges.extend(range(n))
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        i = 0
        while i < len(half_edges):
            j = i + 1
            while j < len(half_edges) and half_edges[j] == half_edges[i]:
                j += 1
            if j >= len(half_edges):
                # Retry
                return random_3_regular(n)
            u, v = half_edges[i], half_edges[j]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                i = j + 1
            else:
                i += 1
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return random_3_regular(n)
        return adj
    
    def edges_from_adj(n, adj):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges
    
    # Try different sizes, starting from Moore bound
    moore_bound = 2 * sum(2**i for i in range(g // 2))
    # moore_bound for g=16: 2*(1+2+4+8+16+32+64+128) = 2*255 = 510
    
    best_n = None
    best_edges = None
    best_girth = 0
    
    # Try sizes near the known best (936) and below
    candidate_sizes = [512, 514, 516, 518, 520, 522, 524, 526, 528, 530,
                       540, 550, 560, 570, 580, 590, 600,
                       650, 700, 750, 800, 850, 900, 936]
    
    # Sort by size
    candidate_sizes = sorted(set(candidate_sizes))
    
    for n in candidate_sizes:
        if time.time() > deadline:
            break
        if n < moore_bound:
            continue
        # n must be even for 3-regular
        if n % 2 != 0:
            n += 1
        
        # Try a few random instances with local search
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            adj = random_3_regular(n)
            if adj is None:
                continue
            
            curr_girth = girth_of_graph(n, adj)
            
            # Local search: edge swaps to increase girth
            improved = True
            max_swaps = 50
            for _ in range(max_swaps):
                if time.time() > deadline:
                    break
                # Find a short cycle and try to break it
                # Try random edge swaps
                swapped = False
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    # Pick random edges to swap
                    all_edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                all_edges.append((i, j))
                    if len(all_edges) < 2:
                        break
                    e1 = random.choice(all_edges)
                    e2 = random.choice(all_edges)
                    if e1 == e2:
                        continue
                    (u1, v1), (u2, v2) = e1, e2
                    # Try swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
                    # or (u1,v2) and (v1,u2)
                    for new1, new2 in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
                        if new1[0] == new1[1] or new2[0] == new2[1]:
                            continue
                        if new1 in adj[new1[0]] or new2 in adj[new2[0]]:
                            continue
                        # Apply swap
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        adj[new1[0]].add(new1[1])
                        adj[new1[1]].add(new1[0])
                        adj[new2[0]].add(new2[1])
                        adj[new2[1]].add(new2[0])
                        
                        new_girth = girth_of_graph(n, adj)
                        if new_girth > curr_girth:
                            curr_girth = new_girth
                            swapped = True
                            break
                        else:
                            # Revert
                            adj[u1].add(v1)
                            adj[v1].add(u1)
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            adj[new1[0]].remove(new1[1])
                            adj[new1[1]].remove(new1[0])
                            adj[new2[0]].remove(new2[1])
                            adj[new2[1]].remove(new2[0])
                    
                    if swapped:
                        break
                if not swapped:
                    break
            
            if curr_girth >= g and n < (best_n if best_n is not None else float('inf')):
                best_n = n
                best_edges = edges_from_adj(n, adj)
                best_girth = curr_girth
                break  # Found valid graph for this n
    
    if best_n is None:
        # Fallback: use LUW construction
        return _luw_construct(k, g)
    
    return best_n, best_edges

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
