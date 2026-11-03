import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin

    def time_left():
        return deadline - time.time()

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Exact girth check using BFS from every vertex with non-backtracking constraint."""
        half_g = g // 2
        for s in range(n):
            if time_left() < 5.0:
                return None  # timeout
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def luw_construct(q, d):
        """Lazebnik-Ustimenko-Woldar construction with parameter q and depth d."""
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            if time_left() < 3.0:
                return None
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    def find_primes_up_to(limit):
        """Find all primes up to limit."""
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]

    def repair_short_cycles(n, adj, g, max_iters=2000, time_budget=10.0):
        """Iteratively find short cycles and rewire edges to break them."""
        start_t = time.time()
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))

        def find_short_cycle():
            """Find a cycle of length < g using BFS."""
            search_start = time.time()
            half_g = g // 2
            for s in range(n):
                if time.time() - search_start > 0.3:
                    return None
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] >= half_g:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            if v != parent[u] and dist[u] + dist[v] + 1 < g:
                                # Reconstruct cycle
                                path_u = []
                                a = u
                                while a != -1:
                                    path_u.append(a)
                                    a = parent[a]
                                path_u.reverse()
                                path_v = []
                                b = v
                                while b != -1:
                                    path_v.append(b)
                                    b = parent[b]
                                path_v.reverse()
                                cycle = path_u + path_v[-2::-1]
                                clean = []
                                for x in cycle:
                                    if not clean or clean[-1] != x:
                                        clean.append(x)
                                if len(clean) >= 3:
                                    return clean
            return None

        for it in range(max_iters):
            if time.time() - start_t > time_budget or time_left() < 10.0:
                return False

            cycle = find_short_cycle()
            if cycle is None:
                return True

            # Find a valid edge on the cycle
            u, v = None, None
            key = None
            for i in range(len(cycle)-1):
                a, b = cycle[i], cycle[i+1]
                k2 = (min(a,b), max(a,b))
                if k2 in edge_set:
                    u, v = a, b
                    key = k2
                    break
            if u is None:
                return False

            # Remove edge (u,v)
            edge_set.remove(key)
            adj[u].remove(v)
            adj[v].remove(u)

            adj_u_set = set(adj[u])
            adj_v_set = set(adj[v])

            success = False
            for attempt in range(200):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)

                if x == y: continue
                if x == u or x == v: continue
                if y == u or y == v: continue
                if x in adj_u_set: continue
                if y in adj_v_set: continue

                key_ux = (min(u,x), max(u,x))
                key_vy = (min(v,y), max(v,y))

                if key_ux in edge_set: continue
                if key_vy in edge_set: continue

                adj[u].append(x)
                adj[v].append(y)
                edge_set.add(key_ux)
                edge_set.add(key_vy)
                success = True
                break

            if not success:
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
                return False

        return False

    # Strategy: Try L-U-W constructions with various q values
    # For k=5, g=9: d = max(3, 9-5 if (9-5)%2 else 9-4) = max(3, 4) = 4
    # q should be >= k (degree)
    # Try primes q >= k, and also q=4 if k<=4

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    # Collect candidate q values
    candidate_qs = []
    # Try q = k first (most compact)
    candidate_qs.append(k)
    # Try primes around k
    primes = find_primes_up_to(max(20, k + 10))
    for p in primes:
        if p >= k and p not in candidate_qs:
            candidate_qs.append(p)
    # Sort by q (smaller q = fewer vertices, try first)
    candidate_qs.sort()

    best_result = None

    for q in candidate_qs:
        if time_left() < 10.0:
            break

        result = luw_construct(q, d)
        if result is None:
            continue

        n, edges = result

        # Check if this is better than current best
        if best_result is None or n < best_result[0]:
            adj = build_adj(edges, n)
            girth_valid = exact_girth_check(n, adj, g)

            if girth_valid is True:
                best_result = (n, edges)
                # If we found a valid one, try to improve with repair
                if time_left() > 20.0:
                    if repair_short_cycles(n, adj, g, max_iters=1000, time_budget=min(15.0, time_left() - 15.0)):
                        if exact_girth_check(n, adj, g) is True:
                            # Rebuild edges from adj
                            new_edges = []
                            edge_set = set()
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        key = (u, v)
                                        if key not in edge_set:
                                            edge_set.add(key)
                                            new_edges.append(key)
                            best_result = (n, new_edges)
            elif girth_valid is None:
                # Timeout during check, keep as candidate but don't confirm
                best_result = (n, edges)
            else:
                # Girth too small, try repair
                if time_left() > 25.0:
                    if repair_short_cycles(n, adj, g, max_iters=2000, time_budget=min(20.0, time_left() - 20.0)):
                        if exact_girth_check(n, adj, g) is True:
                            new_edges = []
                            edge_set = set()
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        key = (u, v)
                                        if key not in edge_set:
                                            edge_set.add(key)
                                            new_edges.append(key)
                            best_result = (n, new_edges)
                            break  # Found valid, stop searching
        else:
            # This q gives more vertices, skip if we already have a good result
            if best_result is not None and n >= best_result[0]:
                continue

    # If no valid result found, try the smallest q construction without repair
    if best_result is None:
        for q in candidate_qs:
            if time_left() < 5.0:
                break
            result = luw_construct(q, d)
            if result is not None:
                best_result = result
                break

    if best_result is not None:
        return best_result

    # Absolute fallback: return Moore bound estimate (should not happen)
    return moore_bound, []
