<<<<<<< SEARCH
def construct(k, g):
    import time
    import random
    from collections import deque

    start_time = time.time()
    deadline = start_time + 135.0
=======
def construct(k, g):
    import time
    import random
    from collections import deque

    start_time = time.time()
    deadline = start_time + 140.0
>>>>>>> REPLACE
<<<<<<< SEARCH
    # ---------- Girth verification (BFS with parent, fast) ----------
    def girth_ok(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True
=======
    # ---------- Girth verification (BFS with parent, fast) ----------
    def girth_ok(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # ---------- Non-backtracking girth check (exact, faster for sparse graphs) ----------
    def nb_girth_ok(n, adj, g):
        """Check girth >= g using non-backtracking walks from every vertex."""
        # For each start vertex, do a BFS on the non-backtracking graph
        # State: (current_vertex, previous_vertex)
        # We track the minimum distance from start to each state
        # A cycle of length L exists iff there's a non-backtracking walk of length L
        # from (v, u) back to (u, v) ... actually simpler:
        # From each vertex s, BFS on non-backtracking walks.
        # If we reach (s, prev) at depth d >= 1, we have a cycle of length d.
        # Actually: start with states (s, v) for each neighbor v of s at depth 1.
        # Then extend. If we reach state (s, w) at depth d, cycle length = d.
        # We need all cycles to have length >= g.
        
        for s in range(n):
            # dist[(u, v)] = length of shortest non-backtracking walk from s to u 
            # that arrived via v (i.e., came from v to u)
            # Initialize: for each neighbor v of s, dist[(s, v)] = 0? No...
            # Let's use: state (u, v) means we are at u, having come from v.
            # Start: for each neighbor v of s, state (v, s) at depth 1.
            # Extend: from (u, v), go to (w, u) for w in adj[u], w != v.
            # A cycle is found when we reach state (s, v) for any v, 
            # meaning we returned to s from v. The cycle length is depth.
            # Actually if we're at state (s, v), that means we came from v to s,
            # and the path from s to s via non-backtracking has length = depth.
            # But we started at s going to v (depth 1 = state (v, s)).
            # So if we reach state (s, v') at depth d, the cycle has length d.
            
            dist = {}
            queue = deque()
            for v in adj[s]:
                dist[(v, s)] = 1
                queue.append((v, s))
            
            while queue:
                u, v = queue.popleft()
                d = dist[(u, v)]
                if d >= g:
                    continue  # No need to extend further
                for w in adj[u]:
                    if w == v:
                        continue  # non-backtracking
                    new_state = (w, u)
                    if new_state in dist:
                        continue
                    dist[new_state] = d + 1
                    if w == s:
                        # Found a cycle of length d + 1
                        if d + 1 < g:
                            return False
                    queue.append(new_state)
        return True
>>>>>>> REPLACE
<<<<<<< SEARCH
    # ---------- Main search ----------
    best_n = None
    best_edges = None

    # Strategy 1: LUW (fast, algebraic, good girth)
    q = k
    if g % 2 == 1:
        start_d = g - 5
        if start_d < 3:
            start_d = 3
        if start_d % 2 == 0:
            start_d += 1
    else:
        start_d = g - 5
        if start_d < 2:
            start_d = 2
        if start_d % 2 == 1:
            start_d += 1

    for d in range(start_d, g + 3):
        if time.time() > deadline:
            break
        if g % 2 == 1 and d % 2 == 0:
            continue
        if g % 2 == 0 and d % 2 == 1:
            continue
        n, edges = build_luw(d, q)
        adj = build_adj(n, edges)
        if girth_ok(n, adj, g):
            best_n, best_edges = n, edges
            break

    # Strategy 2: Cayley graphs for even k
    if k % 2 == 0 and (best_n is None or best_n > 500):
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if time.time() > deadline:
                break
            for d in range(1, 5):
                n = p ** d
                if n < k + 1 or n > 5000:
                    continue
                if (p - 1) % k != 0 and k > 1:
                    continue
                result = build_cayley(p, d, k, 42 + p * 100 + d)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                if time.time() > deadline:
                    break

    # Strategy 3: Random regular bipartite graphs
    if best_n is None or best_n > 1000:
        if g % 2 == 1:
            depth = (g - 3) // 2
        else:
            depth = (g - 4) // 2
        min_n_half = 1
        for i in range(1, depth + 1):
            if i == 1:
                min_n_half += k
            else:
                min_n_half += k * (k - 1) ** (i - 1)

        for n_half in [min_n_half, min_n_half + k, min_n_half + 2 * k, min_n_half + 3 * k]:
            if time.time() > deadline:
                break
            for seed_val in range(50):
                if time.time() > deadline:
                    break
                result = build_random_bipartite(n_half, k, seed_val + 1000)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break

    if best_n is None:
        # Fallback: LUW with largest d
        d = g - 2
        if d < 3:
            d = 3
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges

    return best_n, best_edges
=======
    # ---------- Edge-switching local search to improve girth ----------
    def compute_girth(n, adj):
        """Compute exact girth using non-backtracking BFS from each vertex."""
        best = n + 1
        for s in range(n):
            dist = {}
            queue = deque()
            for v in adj[s]:
                dist[(v, s)] = 1
                queue.append((v, s))
            while queue:
                u, v = queue.popleft()
                d = dist[(u, v)]
                if d >= best:
                    continue
                for w in adj[u]:
                    if w == v:
                        continue
                    ns = (w, u)
                    if ns in dist:
                        continue
                    dist[ns] = d + 1
                    if w == s:
                        if d + 1 < best:
                            best = d + 1
                    queue.append(ns)
        return best if best <= n else n + 1

    def find_short_cycle(n, adj, target_girth):
        """Find a cycle shorter than target_girth, or None."""
        for s in range(n):
            dist = {}
            prev = {}
            queue = deque()
            for v in adj[s]:
                dist[(v, s)] = 1
                prev[(v, s)] = None
                queue.append((v, s))
            while queue:
                u, v = queue.popleft()
                d = dist[(u, v)]
                if d >= target_girth:
                    continue
                for w in adj[u]:
                    if w == v:
                        continue
                    ns = (w, u)
                    if ns in dist:
                        continue
                    dist[ns] = d + 1
                    prev[ns] = (u, v)
                    if w == s:
                        # Reconstruct cycle
                        cycle = []
                        state = ns
                        while state is not None:
                            cycle.append(state[0])
                            state = prev[state]
                        cycle.reverse()
                        # Remove duplicates if any
                        return cycle
                    queue.append(ns)
        return None

    def edge_switching_search(n, adj, edges, k, g, time_limit):
        """Try to increase girth via edge switches. Returns (n, edges) if girth >= g achieved."""
        rng = random.Random(12345)
        current_girth = compute_girth(n, adj)
        if current_girth >= g:
            return n, edges

        # Build edge list as set for quick lookup
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        # Adjacency with edge indices for efficient switching
        # adj[u] = list of (v, edge_index)
        edge_index = {}
        for i, (u, v) in enumerate(edges):
            key = (min(u, v), max(u, v))
            edge_index[key] = i
        
        adj_with_idx = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj_with_idx[u].append((v, i))
            adj_with_idx[v].append((u, i))
        
        best_girth = current_girth
        best_adj = [list(x) for x in adj]
        best_edges = list(edges)
        
        while time.time() < time_limit:
            # Find a short cycle
            cycle = find_short_cycle(n, adj, min(best_girth, g))
            if cycle is None:
                break
            
            # Try to break this cycle via edge switch
            # Pick a random edge on the cycle
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                key = (min(u, v), max(u, v))
                if key in edge_index:
                    cycle_edges.append((u, v, edge_index[key]))
            
            if not cycle_edges:
                break
                
            # Try several random switches
            improved = False
            for _ in range(20):
                u, v, ei = rng.choice(cycle_edges)
                # Remove edge (u, v)
                # Find a valid switch: add (u, w) and (v, z) where 
                # w is neighbor of v (not u), z is neighbor of u (not v),
                # and (u,w) and (v,z) are not existing edges
                candidates_u = [w for w, _ in adj_with_idx[u] if w != v]
                candidates_v = [w for w, _ in adj_with_idx[v] if w != u]
                
                if not candidates_u or not candidates_v:
                    continue
                
                w = rng.choice(candidates_u)
                z = rng.choice(candidates_v)
                
                # Check (v, w) and (u, z) don't already exist
                key_vw = (min(v, w), max(v, w))
                key_uz = (min(u, z), max(u, z))
                if key_vw in edge_set or key_uz in edge_set:
                    continue
                # Also check w != z and the new edges don't create loops
                if w == z:
                    continue
                
                # Perform the switch
                # Remove (u,v), (v,w), (u,z) -- wait, we're removing (u,v) and adding (u,w), (v,z)
                # But (v,w) and (u,z) must not exist
                # Actually: remove edge (u,v). Add edges (u,w) and (v,z).
                # But we need degree to stay k. So we remove (u,v) and also need to 
                # remove one more edge from each of u and v, then add two new ones.
                # Standard 2-switch: pick two edges (u,v) and (w,z), replace with (u,w) and (v,z)
                # or (u,z) and (v,w).
                
                # Let me redo: a 2-switch picks two edges (a,b) and (c,d) and replaces
                # with (a,c) and (b,d) or (a,d) and (b,c), provided no multi-edges.
                
                # So pick another edge (c, d) randomly
                all_edge_list = list(edge_set)
                c, d = rng.choice(all_edge_list)
                if c == u or c == v or d == u or d == v:
                    continue
                
                # Try switch: remove (u,v) and (c,d), add (u,c) and (v,d)
                key_uc = (min(u, c), max(u, c))
                key_vd = (min(v, d), max(v, d))
                if key_uc in edge_set or key_vd in edge_set:
                    # Try other pairing
                    key_ud = (min(u, d), max(u, d))
                    key_vc = (min(v, c), max(v, c))
                    if key_ud in edge_set or key_vc in edge_set:
                        continue
                    # Use (u,d) and (v,c)
                    new_edges = [(u, d), (v, c)]
                    removed = [(u, v), (c, d)]
                else:
                    new_edges = [(u, c), (v, d)]
                    removed = [(u, v), (c, d)]
                
                # Apply switch
                for ru, rv in removed:
                    key = (min(ru, rv), max(ru, rv))
                    del edge_set[key]
                    # Remove from adj_with_idx
                    adj_with_idx[ru].remove((rv, edge_index[key]))
                    adj_with_idx[rv].remove((ru, edge_index[key]))
                    del edge_index[key]
                
                for nu, nv in new_edges:
                    key = (min(nu, nv), max(nu, nv))
                    edge_set.add(key)
                    new_idx = len(edge_index)
                    edge_index[key] = new_idx
                    adj_with_idx[nu].append((nv, new_idx))
                    adj_with_idx[nv].append((nu, new_idx))
                
                # Rebuild adj
                adj = [list(set(x[0] for x in adj_with_idx[i])) for i in range(n)]
                # Actually adj_with_idx has duplicates issues, let's rebuild properly
                adj = [[] for _ in range(n)]
                for i in range(n):
                    seen = set()
                    for nb, _ in adj_with_idx[i]:
                        if nb not in seen:
                            adj[i].append(nb)
                            seen.add(nb)
                
                # Check girth
                new_girth = compute_girth(n, adj)
                if new_girth >= best_girth:
                    best_girth = new_girth
                    best_adj = [list(x) for x in adj]
                    # Rebuild edges from adj
                    best_edges = []
                    seen_e = set()
                    for i in range(n):
                        for j in adj[i]:
                            key = (min(i, j), max(i, j))
                            if key not in seen_e:
                                seen_e.add(key)
                                best_edges.append(key)
                    improved = True
                    if new_girth >= g:
                        return n, best_edges
                else:
                    # Revert
                    for nu, nv in new_edges:
                        key = (min(nu, nv), max(nu, nv))
                        del edge_set[key]
                        idx = edge_index.pop(key)
                        adj_with_idx[nu].remove((nv, idx))
                        adj_with_idx[nv].remove((nu, idx))
                    
                    for ru, rv in removed:
                        key = (min(ru, rv), max(ru, rv))
                        edge_set.add(key)
                        idx = edge_index[key]
                        adj_with_idx[ru].append((rv, idx))
                        adj_with_idx[rv].append((ru, idx))
                    
                    # Rebuild adj
                    adj = [[] for _ in range(n)]
                    for i in range(n):
                        seen = set()
                        for nb, _ in adj_with_idx[i]:
                            if nb not in seen:
                                adj[i].append(nb)
                                seen.add(nb)
        
        return n, best_edges

    # ---------- Main search ----------
    best_n = None
    best_edges = None

    # Strategy 1: LUW (fast, algebraic, good girth)
    q = k
    if g % 2 == 1:
        start_d = g - 5
        if start_d < 3:
            start_d = 3
        if start_d % 2 == 0:
            start_d += 1
    else:
        start_d = g - 5
        if start_d < 2:
            start_d = 2
        if start_d % 2 == 1:
            start_d += 1

    for d in range(start_d, g + 3):
        if time.time() > deadline:
            break
        if g % 2 == 1 and d % 2 == 0:
            continue
        if g % 2 == 0 and d % 2 == 1:
            continue
        n, edges = build_luw(d, q)
        adj = build_adj(n, edges)
        if nb_girth_ok(n, adj, g):
            best_n, best_edges = n, edges
            break

    # Strategy 2: Edge-switching local search on LUW result
    if best_n is not None and best_n > 100 and time.time() < deadline - 20:
        # Try to improve via edge switching
        adj = build_adj(best_n, best_edges)
        cur_girth = compute_girth(best_n, adj)
        if cur_girth < g:
            remaining_time = deadline - 30 - time.time()
            if remaining_time > 5:
                result = edge_switching_search(best_n, adj, best_edges, k, g, time.time() + remaining_time)
                if result is not None:
                    rn, re = result
                    if rn < best_n or (rn == best_n and nb_girth_ok(rn, build_adj(rn, re), g)):
                        best_n, best_edges = rn, re
        else:
            # Already has girth >= g, verify with nb check
            if nb_girth_ok(best_n, adj, g):
                pass  # Good

    # Strategy 3: Cayley graphs for even k
    if k % 2 == 0 and (best_n is None or best_n > 500):
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if time.time() > deadline:
                break
            for d in range(1, 5):
                n = p ** d
                if n < k + 1 or n > 5000:
                    continue
                if (p - 1) % k != 0 and k > 1:
                    continue
                result = build_cayley(p, d, k, 42 + p * 100 + d)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                if time.time() > deadline:
                    break

    # Strategy 4: Random regular bipartite graphs
    if best_n is None or best_n > 1000:
        if g % 2 == 1:
            depth = (g - 3) // 2
        else:
            depth = (g - 4) // 2
        min_n_half = 1
        for i in range(1, depth + 1):
            if i == 1:
                min_n_half += k
            else:
                min_n_half += k * (k - 1) ** (i - 1)

        for n_half in [min_n_half, min_n_half + k, min_n_half + 2 * k, min_n_half + 3 * k]:
            if time.time() > deadline:
                break
            for seed_val in range(50):
                if time.time() > deadline:
                    break
                result = build_random_bipartite(n_half, k, seed_val + 1000)
                if result is not None:
                    n, edges = result
                    if best_n is None or n < best_n:
                        best_n, best_edges = n, edges
                    break

    if best_n is None:
        # Fallback: LUW with largest d
        d = g - 2
        if d < 3:
            d = 3
        n, edges = build_luw(d, q)
        best_n, best_edges = n, edges

    return best_n, best_edges
>>>>>>> REPLACE
