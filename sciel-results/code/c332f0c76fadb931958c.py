import time
import math
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, searches for a small 4-regular graph with girth >= 10.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin before 150s kill
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    best_n = None
    best_edges = None

    # Try to find a graph with n < LUW size (2048)
    # Start with target sizes and search
    target_sizes = [320, 340, 360, 400, 500, 700, 1000, 1500, 2048]
    
    def _build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def _count_short_cycles(n, adj, g):
        """Count cycles of length < g using non-backtracking walks."""
        short_count = 0
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            while queue:
                u, prev, d = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g:
                        short_count += 1
                        if short_count > 1000:  # Early exit
                            return short_count
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, d + 1))
        return short_count

    def _has_girth_at_least(n, edges, g):
        adj = _build_adj(n, edges)
        return _count_short_cycles(n, adj, g) == 0

    def _gen_random_4reg(n):
        """Generates a random 4-regular graph on n vertices."""
        for attempt in range(50):
            stubs = []
            for i in range(n):
                stubs.extend([i] * 4)
            random.shuffle(stubs)
            
            pairs = []
            for i in range(0, len(stubs), 2):
                pairs.append([stubs[i], stubs[i+1]])
            
            # Fix loops
            fixed = True
            for i in range(len(pairs)):
                if pairs[i][0] == pairs[i][1]:
                    u = pairs[i][0]
                    found = False
                    for j in range(len(pairs)):
                        if j == i:
                            continue
                        a, b = pairs[j]
                        if a != u and b != u:
                            pairs[i] = [u, a]
                            pairs[j] = [u, b]
                            found = True
                            break
                    if not found:
                        fixed = False
                        break
                if not fixed:
                    break
            
            if not fixed:
                continue
            
            # Check for multi-edges
            edge_set = set()
            valid = True
            for i in range(len(pairs)):
                u, v = pairs[i]
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
            
            if valid:
                return [(pairs[i][0], pairs[i][1]) for i in range(len(pairs))]
        
        return None

    def _local_search(edges, n, g, time_budget=5.0):
        """Local search to eliminate short cycles via 2-switches."""
        adj = _build_adj(n, edges)
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        start_ls = time.time()
        
        for it in range(20000):
            if time.time() - start_ls > time_budget or time.time() > deadline - 3:
                break
            
            sc = _count_short_cycles(n, adj, g)
            if sc == 0:
                return edges
            
            # Find a short cycle to target
            # Do a 2-switch: pick two edges (u1,v1) and (u2,v2), replace with (u1,u2) and (v1,v2)
            for trial in range(10):
                e1_idx = random.randint(0, len(edges) - 1)
                e2_idx = random.randint(0, len(edges) - 1)
                if e1_idx == e2_idx:
                    continue
                
                u1, v1 = edges[e1_idx]
                u2, v2 = edges[e2_idx]
                
                # Try all 3 possible pairings
                for new_e1, new_e2 in [((u1,u2),(v1,v2)), ((u1,v2),(v1,u2))]:
                    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                        continue
                    ne1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                    ne2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                    
                    if ne1 == ne2:
                        continue
                    
                    old_e1 = (min(u1,v1), max(u1,v1))
                    old_e2 = (min(u2,v2), max(u2,v2))
                    
                    if old_e1 in edge_set and old_e2 in edge_set:
                        edge_set.discard(old_e1)
                        edge_set.discard(old_e2)
                        if ne1 not in edge_set and ne2 not in edge_set:
                            edge_set.add(ne1)
                            edge_set.add(ne2)
                            edges[e1_idx] = new_e1
                            edges[e2_idx] = new_e2
                            adj = _build_adj(n, edges)
                            new_sc = _count_short_cycles(n, adj, g)
                            if new_sc <= sc:
                                sc = new_sc
                                if sc == 0:
                                    return edges
                            else:
                                # Revert
                                edge_set.discard(ne1)
                                edge_set.discard(ne2)
                                edge_set.add(old_e1)
                                edge_set.add(old_e2)
                                edges[e1_idx] = (u1, v1)
                                edges[e2_idx] = (u2, v2)
                                adj = _build_adj(n, edges)
        
        return edges

    # Search loop: try various sizes
    for target_n in target_sizes:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random starts
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            edges = _gen_random_4reg(target_n)
            if edges is None:
                continue
            
            if _has_girth_at_least(target_n, edges, g):
                return target_n, edges
            
            # Try local search
            if time.time() < deadline - 10:
                edges = _local_search(edges, target_n, g)
                if _has_girth_at_least(target_n, edges, g):
                    return target_n, edges
            
            # Track best found so far (even if not girth-valid, for potential use)
            if best_n is None or target_n < best_n:
                adj = _build_adj(target_n, edges)
                sc = _count_short_cycles(target_n, adj, g)
                if sc == 0:
                    best_n = target_n
                    best_edges = edges[:]
                    return target_n, edges

    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
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
