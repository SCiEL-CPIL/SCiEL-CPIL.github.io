import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110

    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            return k * (k - 1) ** (h - 1)
        else:
            h = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(h))

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar construction.
        q: prime power
        d: depth parameter
        Returns (n, edges) for a 3-regular graph.
        """
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
        rule = {2: (1, 1), 3: (2, 1)}
        for c in range(4, d + 1):
            r = (c - 4) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
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
                yield (1 - side, tuple(y))
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def has_girth_at_least(n, adj, min_girth):
        """
        Check if graph has girth >= min_girth using non-backtracking walks.
        Returns True if girth >= min_girth, False otherwise.
        Optimized with early termination.
        """
        for start in range(n):
            # BFS in non-backtracking state space from start
            # State: (current, prev)
            visited = set()
            queue = deque()
            
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            
            while queue:
                u, prev, dist = queue.popleft()
                
                if dist >= min_girth:
                    continue
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False
                    if (nb, u) not in visited:
                        visited.add((nb, u))
                        queue.append((nb, u, dist + 1))
        
        return True

    def find_short_cycle(n, adj, max_len):
        """Find a cycle of length < max_len, return it or None."""
        for start in range(n):
            visited = {}
            queue = deque()
            
            for nb in adj[start]:
                queue.append((nb, start, 1, [start, nb]))
                visited[(nb, start)] = [start, nb]
            
            while queue:
                u, prev, dist, path = queue.popleft()
                
                if dist >= max_len:
                    continue
                
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return path + [start]
                    if (nb, u) not in visited:
                        visited[(nb, u)] = path + [nb]
                        queue.append((nb, u, dist + 1, path + [nb]))
        
        return None

    def random_3reg(n, rng):
        """Generate a random 3-regular graph on n vertices."""
        for _ in range(50):
            stubs = []
            for i in range(n):
                stubs += [i] * 3
            rng.shuffle(stubs)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 3 * n, 2):
                a, b = stubs[i], stubs[i + 1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if ok:
                return [list(s) for s in adj]
        return None

    def fix_cycles(n, adj, min_girth, rng, max_iter=200):
        """Try to fix short cycles by rewiring edges."""
        adj = [list(s) for s in adj]
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            cycle = find_short_cycle(n, adj, min_girth)
            if cycle is None:
                return adj
            
            # Find an edge in the cycle to remove and rewire
            # Pick a random edge from the cycle
            edges_in_cycle = set()
            for i in range(len(cycle) - 1):
                a, b = cycle[i], cycle[i + 1]
                if a < b:
                    edges_in_cycle.add((a, b))
                else:
                    edges_in_cycle.add((b, a))
            
            if not edges_in_cycle:
                break
            
            edge = rng.choice(list(edges_in_cycle))
            u, v = edge
            
            # Remove edge (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Find new neighbors: u needs a new neighbor, v needs a new neighbor
            # u's current neighbors (excluding v)
            u_neighbors = adj[u][:]
            v_neighbors = adj[v][:]
            
            # Find a valid rewiring: connect u to some x not adjacent to u,
            # and v to some y not adjacent to v, where x != y and no multi-edges
            candidates_u = [x for x in range(n) if x != u and x not in adj[u]]
            candidates_v = [y for y in range(n) if y != v and y not in adj[v]]
            
            if not candidates_u or not candidates_v:
                # Restore
                adj[u].append(v)
                adj[v].append(u)
                continue
            
            # Try several rewirings
            for _ in range(20):
                xu = rng.choice(candidates_u)
                yv = rng.choice(candidates_v)
                if xu == yv:
                    continue
                if xu in adj[v] or yv in adj[u]:
                    continue
                if xu == u or yv == v:
                    continue
                
                # Check no multi-edge
                if xu in adj[u] or yv in adj[v]:
                    continue
                
                # Apply rewiring
                adj[u].append(xu)
                adj[xu].append(u)
                adj[v].append(yv)
                adj[yv].append(v)
                
                # Verify degree
                if len(adj[u]) != 3 or len(adj[v]) != 3 or len(adj[xu]) != 3 or len(adj[yv]) != 3:
                    # Undo
                    adj[u].remove(xu)
                    adj[xu].remove(u)
                    adj[v].remove(yv)
                    adj[yv].remove(v)
                    continue
                
                # Check if this created a shorter cycle (quick check)
                new_cycle = find_short_cycle(n, adj, min_girth)
                if new_cycle is None or len(new_cycle) >= len(cycle):
                    return adj
                else:
                    # Undo
                    adj[u].remove(xu)
                    adj[xu].remove(u)
                    adj[v].remove(yv)
                    adj[yv].remove(v)
            
            # If no good rewiring found, restore
            adj[u].append(v)
            adj[v].append(u)
        
        return None

    # Strategy 1: Try LUW with small parameters
    best_result = None
    rng = random.Random(42)
    
    # Try LUW with various (q, d) - smaller d first for smaller n
    luw_candidates = []
    for q in [3, 4, 5]:
        for d in range(5, 11):
            if time.time() > deadline - 30:
                break
            try:
                n, edges = luw_construct(q, d)
                if n <= 20000:
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    # Verify 3-regular
                    if all(len(adj[i]) == 3 for i in range(n)):
                        luw_candidates.append((n, q, d, edges, adj))
            except Exception:
                continue
    
    # Check girth for LUW candidates (smallest first)
    luw_candidates.sort(key=lambda x: x[0])
    for n, q, d, edges, adj in luw_candidates:
        if time.time() > deadline - 20:
            break
        if has_girth_at_least(n, adj, g):
            best_result = (n, edges)
            break
    
    # Strategy 2: Random search with cycle fixing for smaller n
    if best_result is None:
        mb = moore_bound(k, g)
        for n_try in range(max(mb, 512), 2000, 16):
            if time.time() > deadline - 15:
                break
            if n_try % 2 != 0:
                continue
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                adj = random_3reg(n_try, rng)
                if adj is None:
                    continue
                # Quick girth check
                if has_girth_at_least(n_try, adj, g):
                    edges = []
                    for i in range(n_try):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    best_result = (n_try, edges)
                    break
            if best_result is not None:
                break
    
    # Strategy 3: If still nothing, use LUW fallback
    if best_result is None:
        # Try a few more LUW params
        for q in [3, 5, 7]:
            for d in range(6, 12):
                if time.time() > deadline - 10:
                    break
                try:
                    n, edges = luw_construct(q, d)
                    if n <= 30000:
                        adj = [[] for _ in range(n)]
                        for u, v in edges:
                            adj[u].append(v)
                            adj[v].append(u)
                        if all(len(adj[i]) == 3 for i in range(n)):
                            if has_girth_at_least(n, adj, g):
                                best_result = (n, edges)
                                break
                except Exception:
                    continue
            if best_result is not None:
                break
    
    # Final fallback: LUW with q=3, d=10
    if best_result is None:
        n, edges = luw_construct(3, 10)
        best_result = (n, edges)
    
    return best_result
