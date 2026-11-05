import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 140s limit to be safe
    
    # Moore bound calculation
    # n >= 1 + k * sum_{j=0}^{floor((g-3)/2)} (k-1)^j
    # For g=16, k=3: 1 + 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 382
    max_depth = (g - 3) // 2
    moore_bound = 1
    term = 1
    for j in range(max_depth + 1):
        moore_bound += k * term
        term *= (k - 1)
        
    # Ensure n is even for k=3 circulant (Möbius ladder)
    if k == 3:
        if moore_bound % 2 != 0:
            moore_bound += 1
            
        # Search for the smallest even n >= moore_bound such that girth(M_n) >= g
        # M_n is defined on Z_n with edges (i, i+1), (i, i-1), (i, i+n/2)
        
        def check_girth_mobius_ladder(n, min_girth):
            """
            Returns True if the Möbius ladder M_n has girth >= min_girth.
            Uses BFS from each vertex to find shortest cycle.
            """
            # BFS from each vertex
            # We can optimize: if we find a cycle of length L < min_girth, return False.
            # Cycle length in BFS: dist[u] + dist[v] + 1 where (u,v) is an edge and u,v are in same BFS tree.
            # Actually, standard girth algorithm:
            # For each edge (u,v), remove it, BFS u to v. If dist < min_girth - 1, fail.
            # This is O(E * (V+E)).
            
            # Alternative: BFS from each vertex s.
            # Maintain dist[s][v] and parent.
            # If we encounter an edge (u, v) where v is already visited and v is not parent[u],
            # and dist[u] + dist[v] + 1 < min_girth, then we found a short cycle.
            
            adj = [[] for _ in range(n)]
            h = n // 2
            for i in range(n):
                adj[i].append((i + 1) % n)
                adj[i].append((i - 1) % n)
                adj[i].append((i + h) % n)
            
            # We only need to check cycles up to length min_girth - 1
            # BFS depth limit: min_girth // 2
            
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                
                for u in queue:
                    if dist[u] * 2 + 1 >= min_girth:
                        # Any cycle through u and a neighbor v would have length >= 2*dist[u] + 1
                        # If this is >= min_girth, we don't need to explore deeper from u
                        # for the purpose of finding a cycle SHORTER than min_girth.
                        # Wait, a cycle could go u->...->v->u. Length dist[u->v] + 1.
                        # If we stop expanding u, we might miss a cycle that goes u->w->...->v->u
                        # where dist[u->w] is small but path w->...->v is long?
                        # No, BFS finds shortest paths.
                        # The shortest cycle containing edge (u,v) where v is visited
                        # has length dist[u] + dist[v] + 1.
                        # If dist[u] + dist[v] + 1 < min_girth, we fail.
                        # If dist[u] is large, dist[v] must be small for the sum to be small?
                        # Actually, if dist[u] >= (min_girth - 1) / 2, and dist[v] >= 1,
                        # then dist[u] + dist[v] + 1 >= (min_girth - 1)/2 + 2.
                        # For min_girth=16, (15)/2 + 2 = 9.5. Not necessarily >= 16.
                        # So we cannot prune based solely on dist[u] being half the girth.
                        # We must check all edges.
                        pass
                        
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # v is already visited
                            # If v is parent of u, it's the tree edge, ignore (backtracking)
                            if v == parent[u]:
                                continue
                            # Cycle found: u->v edge, path s->u, path s->v
                            # Length = dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            return True

        n = moore_bound
        while n < 5000: # Upper bound search
            if time.time() > deadline:
                break
            if check_girth_mobius_ladder(n, g):
                # Construct edges
                edges = []
                h = n // 2
                for i in range(n):
                    edges.append((i, (i + 1) % n))
                    edges.append((i, (i - 1) % n))
                    edges.append((i, (i + h) % n))
                return n, edges
            n += 2
            
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    # This is the seed code, kept for other k values or if search fails
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
