import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def moore_bound(k, g):
        if k <= 1: return 0
        if g == 1: return 1
        r = g // 2
        if g % 2 == 1:
            # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)
    
    def action_on_point(x, a, b, c, d, q):
        # x is in 0..q-1 or infinity (represented by q)
        # M = [[a, b], [c, d]]
        # y = (ax + b) / (cx + d)
        if x == q: # infinity
            if c == 0:
                return q
            else:
                # a/c
                inv_c = pow(c, -1, q)
                return (a * inv_c) % q
        else:
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            else:
                inv_den = pow(den, -1, q)
                return (num * inv_den) % q

    def build_and_check(q, k, g):
        # Generate k random matrices in PGL(2, q)
        # We need ad - bc != 0.
        
        for attempt in range(300):
            if time.time() - start_time > timeout:
                return None
            
            gens = []
            valid = True
            for _ in range(k):
                # Random a,b,c,d with det != 0
                while True:
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(0, q - 1)
                    d = random.randint(0, q - 1)
                    det = (a * d - b * c) % q
                    if det != 0:
                        break
                gens.append((a, b, c, d))
            
            # Build graph
            n = q + 1
            adj = [set() for _ in range(n)]
            is_simple = True
            
            for i in range(n):
                x = i if i < q else q
                for gen in gens:
                    y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                    j = y if y < q else q
                    if i == j:
                        is_simple = False
                        break
                    if j in adj[i]:
                        # Multi-edge
                        is_simple = False
                        break
                    adj[i].add(j)
                    adj[j].add(i)
                if not is_simple:
                    break
            
            if not is_simple:
                continue
                
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    is_simple = False
                    break
            
            if not is_simple:
                continue
                
            # Check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 0
            while queue:
                u = queue.popleft()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            if count != n:
                continue
                
            # Check girth using non-backtracking walks
            # For each vertex, do BFS with state (current, prev) to avoid backtracking
            min_girth = float('inf')
            for u in range(n):
                # State: (current_vertex, previous_vertex)
                # Start from (u, None) but we need to avoid going back
                # Use BFS where state is (v, parent)
                dist = {}
                queue_nb = deque()
                # Initialize: from u, go to each neighbor v with state (v, u)
                for v in adj[u]:
                    dist[(v, u)] = 1
                    queue_nb.append((v, u))
                
                found_cycle = False
                while queue_nb:
                    v, parent = queue_nb.popleft()
                    d = dist[(v, parent)]
                    # Check if we've returned to u
                    if v == u:
                        # This would mean we went u -> ... -> u, but we started from u's neighbors
                        # Actually, we want to find a cycle containing u
                        # If we reach a neighbor of u, that forms a cycle
                        pass
                    for w in adj[v]:
                        if w == parent:
                            continue  # Don't backtrack
                        if (w, v) not in dist:
                            dist[(w, v)] = d + 1
                            queue_nb.append((w, v))
                        # Check if w is a neighbor of u (and w != parent to avoid trivial)
                        # If w is in adj[u] and w != parent, we have a cycle u->...->w->u
                        if w in adj[u] and w != parent:
                            cycle_len = d + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            found_cycle = True
                # If no cycle found through this method, there might be none or we need different approach
                
            # Alternative girth check: BFS from each node to find shortest path to any neighbor
            current_girth = float('inf')
            for u in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[u] = 0
                q_bfs = deque([u])
                shortest = float('inf')
                while q_bfs:
                    x = q_bfs.popleft()
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            parent[y] = x
                            q_bfs.append(y)
                        elif y != parent[x] and y != u:
                            # Found a cycle
                            # Cycle length = dist[x] + dist[y] + 1
                            cycle_len = dist[x] + dist[y] + 1
                            if cycle_len < shortest:
                                shortest = cycle_len
                        elif y == u and x != parent[u]:
                            # Back to start, but this is the trivial edge
                            pass
                if shortest < current_girth:
                    current_girth = shortest
            
            if current_girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    # Fallback: L-U-W construction
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

    best_result = None
    best_n = float('inf')
    
    # Try Morgenstern for prime q only (not all prime powers, since modular inverse needs field)
    max_q_to_check = max(500, n_min + 100)
    
    for curr_q in range(max(2, n_min - 1), max_q_to_check + 1):
        if time.time() - start_time > timeout:
            break
        
        # Only try prime q for reliable modular arithmetic
        if not is_prime(curr_q):
            continue
            
        res = build_and_check(curr_q, k, g)
        if res:
            n, edges = res
            if n < best_n:
                best_n = n
                best_result = res
        
        if time.time() - start_time > timeout:
            break

    if best_result:
        return best_result
    else:
        # Fallback to L-U-W
        return luw_construct(k, g)

# EVOLVE-BLOCK-END
