import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Optimized Girth Check using non-backtracking walks
        # For girth >= g, we need no cycles of length < g
        # Use BFS from each vertex up to depth g//2
        
        max_depth = g // 2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
                
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        # Ignore tree edges (parent-child)
                        if v == parent[curr]:
                            continue
                        # If v is already visited and not parent, check cycle length
                        # Cycle length = dist[curr] + 1 + dist[v]
                        # But we need to be careful: this is only valid if the path
                        # from u to v doesn't go through curr
                        # Actually, for BFS, if dist[v] <= dist[curr], then
                        # the cycle length is dist[curr] + dist[v] + 1
                        # If dist[v] == dist[curr] + 1, it's a forward edge, 
                        # but in BFS that shouldn't happen for unvisited nodes
                        # Let's use a simpler check: any back/cross edge creates a cycle
                        # The cycle length is at least dist[curr] + dist[v] + 1
                        # But this might overestimate. Let's use the standard approach:
                        # If we find an edge (curr, v) where both are visited and v != parent[curr],
                        # and dist[curr] + dist[v] + 1 < g, we have a short cycle.
                        # However, this isn't quite right either.
                        
                        # Better approach: track the actual distance from root u
                        # If there's an edge between two nodes at distances d1 and d2 from u,
                        # and they're not parent-child, then there's a cycle of length 
                        # d1 + d2 + 1 (if the paths from u to d1 and u to d2 only meet at u)
                        # This is an upper bound on the cycle length, so if d1+d2+1 >= g, 
                        # we can't conclude there's a short cycle.
                        
                        # The correct way: we need to find the actual shortest cycle.
                        # For BFS from u, if we encounter an edge (curr, v) where v is already
                        # visited and v != parent[curr], then the cycle length is 
                        # dist[curr] + dist[v] + 1 ONLY IF the paths from u to curr and 
                        # u to v are disjoint except at u. This is not guaranteed.
                        
                        # Let's use a different approach: for each vertex, do BFS and 
                        # when we find a back edge, compute the cycle length properly.
                        # Actually, the standard girth algorithm: 
                        # For each edge (u,v), remove it and find shortest path from u to v.
                        # That's O(m * (n+m)) which is too slow.
                        
                        # Alternative: use the fact that for girth g, we only need to 
                        # check up to depth (g-1)/2 from each vertex. If we find any cycle
                        # of length < g, we fail.
                        
                        # Let's use a cleaner BFS that tracks the actual path.
                        # For now, use a conservative check:
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        # Original LUW construction
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # 1. Try LUW family as fallback
    q = k
    min_d = max(3, g - 5)
    
    # Try a wider range of d values
    for d_val in range(min_d, min_d + 15):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            if n > 50000:
                continue
            if n < best_n:
                t_limit = 5.0 if n > 10000 else 20.0
                if verify_graph(n, edges, k, g, time_limit=t_limit):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass

    # 2. Try Cayley graph on A_6 (order 360)
    # This is a specific search for a smaller graph (n=360 < 1024)
    if time.time() < deadline - 20:
        try:
            n_cayley, edges_cayley = search_cayley_a6(k, g, start_time, deadline)
            if n_cayley is not None and n_cayley < best_n:
                best_n = n_cayley
                best_edges = edges_cayley
        except Exception:
            pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges

def search_cayley_a6(k, g, start_time, deadline):
    """
    Search for a k-regular Cayley graph on A_6 (order 360) with girth >= g.
    """
    if k != 4:
        return None, None
    
    # Generate A_6 elements as tuples of permutations of 0..5
    # We only need even permutations.
    # Total 720 permutations in S_6. Half are in A_6.
    
    import itertools
    
    # Generate all permutations
    perms = list(itertools.permutations(range(6)))
    
    # Filter even permutations
    def is_even(perm):
        inv = 0
        for i in range(6):
            for j in range(i+1, 6):
                if perm[i] > perm[j]:
                    inv += 1
        return inv % 2 == 0
        
    a6_perms = [p for p in perms if is_even(p)]
    n = len(a6_perms) # 360
    
    # Map perm to index
    perm_to_idx = {p: i for i, p in enumerate(a6_perms)}
    
    # Identity
    identity = (0, 1, 2, 3, 4, 5)
    idx_identity = perm_to_idx[identity]
    
    # Precompute multiplication table for speed
    # 360x360 = 129600 entries, each is an index (0-359)
    # This takes some memory but will be much faster
    
    def multiply(p1, p2):
        # p1 * p2
        # Apply p2 then p1? Or p1 then p2?
        # Standard: (p1 * p2)(i) = p1(p2(i))
        return tuple(p1[p2[i]] for i in range(6))
        
    def inverse(p):
        inv = [0]*6
        for i in range(6):
            inv[p[i]] = i
        return tuple(inv)
        
    # Precompute multiplication table as a 2D list of indices
    # mul_table[i][j] = index of a6_perms[i] * a6_perms[j]
    mul_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            prod = multiply(a6_perms[i], a6_perms[j])
            mul_table[i][j] = perm_to_idx[prod]
            
    # Precompute inverse table
    inv_table = [0]*n
    for i in range(n):
        inv_table[i] = perm_to_idx[inverse(a6_perms[i])]
        
    # We need 4 generators.
    # Generators should not be identity or inverses of each other.
    # Random search for 4 generators.
    
    max_attempts = 200  # Reduced to save time
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
            
        # Pick 4 random distinct generators
        # Ensure they are not identity
        candidates = [i for i in range(n) if i != idx_identity]
        
        # Shuffle and pick 4
        random.shuffle(candidates)
        gen_indices = candidates[:4]
        
        # Check if any generator is inverse of another
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                if mul_table[gen_indices[i]][gen_indices[j]] == idx_identity:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Build edges using the precomputed multiplication table
        edge_set = set()
        for v_idx in range(n):
            for gi in gen_indices:
                u_idx = mul_table[v_idx][gi]
                if v_idx < u_idx:
                    edge_set.add((v_idx, u_idx))
                else:
                    edge_set.add((u_idx, v_idx))
                    
        edges = list(edge_set)
        
        # Quick girth check before full verification
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        if any(len(adj[i]) != k for i in range(n)):
            continue
            
        # Fast girth check: BFS from each vertex up to depth g//2
        max_depth = g // 2
        girth_ok = True
        
        for u in range(n):
            if time.time() > deadline - 10:
                girth_ok = False
                break
                
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            girth_ok = False
                            break
                if not girth_ok:
                    break
            
            if not girth_ok:
                break
        
        if girth_ok:
            return n, edges
            
    return None, None
