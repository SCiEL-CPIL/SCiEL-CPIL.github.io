import time
import random
import numpy as np
from collections import deque

def _verify(n, edges, k, g):
    """Verify: simple, k-regular, connected, girth >= g via non-backtracking BFS."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False
    # girth via non-backtracking walks from every vertex
    for s in range(n):
        # dist[prev][cur] = distance from s to cur via prev
        dist = {}
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq = deque([(s, nb)])
            while dq:
                prev, cur = dq.popleft()
                d = dist[(prev, cur)]
                if d >= g - 1:
                    continue
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == s:
                        if d + 1 < g:
                            return False
                        continue
                    key = (cur, nb)
                    if key not in dist:
                        dist[key] = d + 1
                        dq.append((cur, nb))
    return True

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g using non-backtracking BFS.
    Returns (length, edge_set) or (None, None)."""
    for s in range(n):
        dist = {}
        for nb in adj[s]:
            dist[(s, nb)] = 1
            dq = deque([(s, nb, [nb])])
            while dq:
                prev, cur, path = dq.popleft()
                d = dist[(prev, cur)]
                if d >= g - 1:
                    continue
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if nb == s:
                        clen = d + 1
                        if clen < g:
                            return clen, path
                        continue
                    key = (cur, nb)
                    if key not in dist:
                        dist[key] = d + 1
                        dq.append((cur, nb, path + [nb]))
    return None, None

def _build_luw(k, g):
    """Construct Lazebnik-Ustimenko-Woldar graph as fallback."""
    q = k
    if q == 4:
        d = 5 if g <= 10 else 6
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def _add(a,b): return a ^ b
        def _sub(a,b): return a ^ b
        def _mul(a,b): return tbl[a][b]
    else:
        d = max(3, g - 5 if (g-5) % 2 else g - 4)
        def _add(a,b): return (a+b) % q
        def _sub(a,b): return (a-b) % q
        def _mul(a,b): return (a*b) % q
    
    rules = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5) % 4
        c0 = c - r
        rules[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def _nbrs(v):
        side, x = v
        for f in range(q):
            y = [None, f] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rules[c]
                if side == 0:
                    y[c] = _add(x[c], _mul(y[a], x[b]))
                else:
                    y[c] = _sub(x[c], _mul(x[a], y[b]))
            yield (1-side, tuple(y))
    
    start = (0, (None,) + (0,)*d)
    idx = {start: 0}
    ordered = [start]
    edge_list = []
    for v in ordered:
        for nb in _nbrs(v):
            if nb not in idx:
                idx[nb] = len(ordered)
                ordered.append(nb)
            if v[0] == 0:
                edge_list.append((idx[v], idx[nb]))
    return len(ordered), edge_list

def _gen_4reg(n):
    """Generate a random simple 4-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != 4:
            return None
    return adj

def construct(k, g):
    """Construct k-regular graph with girth at least g, minimizing vertices."""
    t0 = time.time()
    hard_limit = t0 + 115  # Leave 5s buffer for 120s limit
    
    if k == 4 and g == 10:
        # Strategy: iterative improvement via cycle-breaking swaps on a random 4-reg graph
        # Start with random 4-reg graph on n vertices, repeatedly find short cycles
        # and break them via 2-switches (edge swaps). Track best n found.
        
        best_n = 320
        best_edges = None
        
        for n in [320, 322, 324, 326, 328, 330, 332, 334, 336]:
            if time.time() > hard_limit - 10:
                break
            
            n_deadline = time.time() + 12
            # Try multiple random starting graphs
            for seed_try in range(30):
                if time.time() > n_deadline:
                    break
                if time.time() > hard_limit - 5:
                    break
                
                adj = _gen_4reg(n)
                if adj is None:
                    continue
                
                # Check if already valid
                clen, _ = _find_short_cycle(adj, n, g)
                if clen is None:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if _verify(n, edges, 4, g):
                        return n, edges
                
                # Iterative cycle-breaking
                max_steps = 500
                for step in range(max_steps):
                    if time.time() > n_deadline or time.time() > hard_limit - 5:
                        break
                    
                    clen, cycle_path = _find_short_cycle(adj, n, g)
                    if clen is None:
                        # No short cycles found
                        edges = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        if _verify(n, edges, 4, g):
                            return n, edges
                        break
                    
                    # Try to break the cycle via 2-switches
                    found_break = False
                    # Get the cycle edges
                    # cycle_path contains vertices in the cycle (from s's neighbor back toward s)
                    # The cycle is: s -> cycle_path[0] -> cycle_path[1] -> ... -> s
                    # Actually cycle_path is the path from s through non-backtracking
                    # Let me reconstruct: the cycle is s, then the path
                    # We need to find an edge on the cycle and swap it with an off-cycle edge
                    
                    # Find all edges on the cycle
                    cycle_edges = set()
                    # Reconstruct cycle from path
                    # The path in _find_short_cycle: when we find nb == s, the path is [v1, v2, ..., vk]
                    # and the cycle is s -> v1 -> v2 -> ... -> vk -> s
                    # So cycle edges are (s,v1), (v1,v2), ..., (vk, s)
                    if cycle_path:
                        full_cycle = [s] + list(cycle_path)
                        for i in range(len(full_cycle)):
                            u1 = full_cycle[i]
                            v1 = full_cycle[(i+1) % len(full_cycle)]
                            if u1 < v1:
                                cycle_edges.add((u1, v1))
                            else:
                                cycle_edges.add((v1, u1))
                        
                        # Try to break by 2-switching a cycle edge
                        for _attempt in range(20):
                            if time.time() > n_deadline or time.time() > hard_limit - 5:
                                break
                            # Pick a random cycle edge
                            ce = random.choice(list(cycle_edges))
                            u, v = ce
                            # Find neighbors of u and v other than each other
                            nbrs_u = [x for x in adj[u] if x != v]
                            nbrs_v = [y for y in adj[v] if y != u]
                            if not nbrs_u or not nbrs_v:
                                continue
                            x = random.choice(nbrs_u)
                            y = random.choice(nbrs_v)
                            
                            # Check swap validity: new edges (u,y) and (v,x) must not exist
                            if x == y:
                                continue
                            if y in adj[u]:  # (u,y) already exists
                                continue
                            if x in adj[v]:  # (v,x) already exists
                                continue
                            # Also ensure we don't create a loop
                            if x == u or y == v:
                                continue
                            
                            # Perform the 2-switch
                            adj[u].remove(v)
                            adj[v].remove(u)
                            adj[u].append(y)
                            adj[y].append(u)
                            adj[v].append(x)
                            adj[x].append(v)
                            found_break = True
                            break
                    
                    if not found_break:
                        # Try random swaps elsewhere
                        for _attempt in range(10):
                            if time.time() > n_deadline or time.time() > hard_limit - 5:
                                break
                            u = random.randrange(n)
                            v = random.choice(adj[u])
                            nbrs_u = [x for x in adj[u] if x != v]
                            nbrs_v = [y for y in adj[v] if y != u]
                            if not nbrs_u or not nbrs_v:
                                continue
                            x = random.choice(nbrs_u)
                            y = random.choice(nbrs_v)
                            if x == y or x in adj[v] or y in adj[u]:
                                continue
                            if x == u or y == v:
                                continue
                            adj[u].remove(v)
                            adj[v].remove(u)
                            adj[u].append(y)
                            adj[y].append(u)
                            adj[v].append(x)
                            adj[x].append(v)
                            found_break = True
                            break
        
        if best_edges is not None:
            return best_n, best_edges
        else:
            # Fallback to LUW construction
            return _build_luw(k, g)
    else:
        return _build_luw(k, g)
