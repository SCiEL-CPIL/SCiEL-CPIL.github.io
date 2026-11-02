"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        elif k == 3:
            moore = 2 * (3 ** ((g - 1) // 2))
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check simple: no self-loops, no duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # BFS from each vertex to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def compute_girth(n, adj):
        """Compute exact girth of graph."""
        if n == 0:
            return float('inf')
        girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
        return girth

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def local_search_improve(n, edges, target_girth, max_time=30.0):
        """Improve girth via edge swaps while maintaining k-regularity."""
        start_ls = time.time()
        adj = build_adj(n, edges)
        current_girth = compute_girth(n, adj)
        
        if current_girth >= target_girth:
            return n, edges, current_girth
        
        # Build edge set for fast lookup
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        # Find all edges that participate in short cycles
        short_edges = set()
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            # Mark edges on this cycle
                            # Reconstruct cycle
                            path1 = []
                            x = u
                            while x != -1:
                                path1.append(x)
                                x = parent[x]
                            path2 = []
                            x = v
                            while x != -1:
                                path2.append(x)
                                x = parent[x]
                            # Find LCA
                            set1 = set(path1)
                            lca = None
                            for node in path2:
                                if node in set1:
                                    lca = node
                                    break
                            if lca is not None:
                                # Cycle edges
                                i = path1.index(lca)
                                j = path2.index(lca)
                                cycle_nodes = path1[i:] + path2[j:][::-1]
                                for a in range(len(cycle_nodes)):
                                    b = (a + 1) % len(cycle_nodes)
                                    e = (min(cycle_nodes[a], cycle_nodes[b]), 
                                         max(cycle_nodes[a], cycle_nodes[b]))
                                    short_edges.add(e)
        
        if not short_edges:
            return n, edges, current_girth
        
        # Try to swap edges to break short cycles
        for iteration in range(10000):
            if time.time() - start_ls > max_time:
                break
            if time_left() < 10:
                break
            
            # Pick a random short edge to remove
            if not short_edges:
                break
            e_remove = random.choice(list(short_edges))
            u, v = e_remove
            
            # Find candidate edges to add that maintain k-regularity
            # We need to find w, x such that (u,w) and (v,x) are non-edges
            # and swapping (u,v) -> (u,w)+(v,x) minus something... 
            # Actually for edge swap in regular graph: remove (u,v), add (u,w) and (v,x) 
            # but that changes degrees. Instead: remove (u,v) and (w,x), add (u,w) and (v,x)
            # or remove (u,v) and (w,x), add (u,x) and (v,w)
            
            # Simpler: try 2-opt style swaps
            # Remove edge (u,v), find non-neighbors w of u and x of v
            # such that (u,w) and (v,x) are not edges, and w != x
            # Then we'd need to also fix degrees... 
            # Actually let's do: remove (u,v), add (u,w) where w is non-neighbor of u
            # But that makes deg(u) same, deg(v) decrease by 1, deg(w) increase by 1
            # So we need to also remove an edge from v and add to w...
            
            # Better approach: edge swap (2-opt)
            # Find edges (u,a) and (v,b) where a,b are neighbors
            # Remove (u,a) and (v,b), add (u,b) and (v,a)
            # This maintains k-regularity if (u,b) and (v,a) are not existing edges
            
            candidates = []
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b == u:
                        continue
                    if (min(u, b), max(u, b)) in edge_set:
                        continue
                    if (min(v, a), max(v, a)) in edge_set:
                        continue
                    candidates.append((a, b))
            
            if not candidates:
                short_edges.discard(e_remove)
                continue
            
            a, b = random.choice(candidates)
            
            # Perform swap: remove (u,a) and (v,b), add (u,b) and (v,a)
            new_edges = []
            for e in edges:
                eu, ev = e
                ek = (min(eu, ev), max(eu, ev))
                if ek == (min(u, a), max(u, a)):
                    continue
                if ek == (min(v, b), max(v, b)):
                    continue
                new_edges.append(e)
            new_edges.append((u, b))
            new_edges.append((v, a))
            
            # Verify and check girth improvement
            new_adj = build_adj(n, new_edges)
            new_girth = compute_girth(n, new_adj)
            
            if new_girth >= current_girth:
                edges = new_edges
                adj = new_adj
                current_girth = new_girth
                edge_set = set()
                for eu, ev in edges:
                    edge_set.add((min(eu, ev), max(eu, ev)))
                # Recompute short edges
                short_edges = set()
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    q = deque([start])
                    dist[start] = 0
                    while q:
                        u2 = q.popleft()
                        for v2 in adj[u2]:
                            if dist[v2] == -1:
                                dist[v2] = dist[u2] + 1
                                parent[v2] = u2
                                q.append(v2)
                            elif v2 != parent[u2] and u2 != parent[v2]:
                                cycle_len = dist[u2] + dist[v2] + 1
                                if cycle_len < target_girth:
                                    path1 = []
                                    x = u2
                                    while x != -1:
                                        path1.append(x)
                                        x = parent[x]
                                    path2 = []
                                    x = v2
                                    while x != -1:
                                        path2.append(x)
                                        x = parent[x]
                                    set1 = set(path1)
                                    lca = None
                                    for node in path2:
                                        if node in set1:
                                            lca = node
                                            break
                                    if lca is not None:
                                        i = path1.index(lca)
                                        j = path2.index(lca)
                                        cycle_nodes = path1[i:] + path2[j:][::-1]
                                        for ci in range(len(cycle_nodes)):
                                            cj = (ci + 1) % len(cycle_nodes)
                                            e = (min(cycle_nodes[ci], cycle_nodes[cj]), 
                                                 max(cycle_nodes[ci], cycle_nodes[cj]))
                                            short_edges.add(e)
                
                if current_girth >= target_girth:
                    break
        
        return n, edges, current_girth

    def generate_random_regular(n, k_val, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        stubs = [i for i in range(n) for _ in range(k_val)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(n * k_val // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edges = list(edge_set)
        return n, edges

    def try_explicit_construction():
        """Try the Lazebnik-Ustimenko-Woldar construction."""
        q = k
        # For L-U-W, we need q to be a prime power and girth = d+1 where d is the length
        # The construction gives girth d+1 for odd d >= 3
        # For girth g (odd), we need d = g-1
        d = g - 1
        if d < 3:
            return None
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Rule for the construction
        rule = {}
        for c in range(2, d + 1):
            if c == 2:
                rule[c] = (1, 1)
            elif c == 3:
                rule[c] = (2, 1)
            elif c == 4:
                rule[c] = (1, 2)
            else:
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
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
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    # Step 1: Try explicit construction
    if time_left() > 15:
        try:
            result = try_explicit_construction()
            if result is not None:
                n_exp, edges_exp = result
                if verify_girth(n_exp, edges_exp, g):
                    best_n = n_exp
                    best_edges = edges_exp
        except Exception:
            pass

    # Step 2: If no valid explicit construction, try to find one via search
    if best_n is None or time_left() > 20:
        # Try various sizes near the Moore bound and above
        start_n = max(moore, k + 1)
        
        if best_n is not None:
            search_hi = best_n
        else:
            search_hi = min(start_n * 5, 500)
        
        lo = start_n
        hi = search_hi
        
        # Try a range of n values
        n_values = list(range(lo, hi + 1, max(1, (hi - lo) // 15)))
        
        for n_try in n_values:
            if time_left() < 15:
                break
            
            for attempt in range(30):
                if time_left() < 15:
                    break
                result = generate_random_regular(n_try, k, seed=random.randint(0, 2**31))
                if result is not None:
                    n_r, edges_r = result
                    adj_r = build_adj(n_r, edges_r)
                    girth_r = compute_girth(n_r, adj_r)
                    
                    if girth_r >= g:
                        if best_n is None or n_r < best_n:
                            best_n = n_r
                            best_edges = edges_r
                        break
                    else:
                        # Try local search to improve girth
                        if time_left() > 25:
                            n_ls, edges_ls, girth_ls = local_search_improve(
                                n_r, edges_r, g, max_time=min(15.0, time_left() - 15)
                            )
                            if girth_ls >= g:
                                if best_n is None or n_ls < best_n:
                                    best_n = n_ls
                                    best_edges = edges_ls
                                break

    # Step 3: If still no result, try more aggressive search
    if best_n is None and time_left() > 10:
        # Try a few more sizes with local search
        for n_try in range(max(moore, k + 1), max(moore, k + 1) + 100, 5):
            if time_left() < 5:
                break
            for attempt in range(10):
                if time_left() < 5:
                    break
                result = generate_random_regular(n_try, k, seed=random.randint(0, 2**31))
                if result is not None:
                    n_r, edges_r = result
                    if time_left() > 10:
                        n_ls, edges_ls, girth_ls = local_search_improve(
                            n_r, edges_r, g, max_time=min(8.0, time_left() - 5)
                        )
                        if girth_ls >= g:
                            if best_n is None or n_ls < best_n:
                                best_n = n_ls
                                best_edges = edges_ls
                            break

    if best_n is None:
        # Fallback: return explicit construction even if girth not verified
        try:
            result = try_explicit_construction()
            if result is not None:
                return result
        except Exception:
            pass
        
        # Last resort: return a random regular graph
        n_fb = max(moore, k + 1)
        for attempt in range(100):
            result = generate_random_regular(n_fb, k, seed=random.randint(0, 2**31))
            if result is not None:
                return result
        
        # Absolute fallback
        return n_fb, []

    return best_n, best_edges
# EVOLVE-BLOCK-END
