"""Rewrite for: dense triangle-free graphs on 16 vertices, chi >= 4, maximize edges."""
import time
import random
import sys

def _is_triangle_free(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                if adj[i] & adj[j]:
                    return False
    return True

def _edge_count(adj, n):
    c = 0
    for i in range(n):
        c += bin(adj[i]).count('1')
    return c // 2

def _can_add(adj, u, v):
    if adj[u] & (1 << v):
        return False
    return (adj[u] & adj[v]) == 0

def _add(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _is_3_colorable(adj, n, time_limit=None):
    """
    DSATUR-based 3-colorability check with bitmasks.
    Returns True if 3-colorable, False if not, None on timeout.
    """
    if time_limit is None:
        time_limit = 1.0
    
    start_time = time.time()
    adj_list = [adj[i] for i in range(n)]
    
    # DSATUR: select vertex with max saturation (distinct neighbor colors), tie-break by degree
    colors = [-1] * n
    sat = [0] * n  # bitmask of colors used by neighbors
    
    # Precompute degrees
    degs = [bin(adj[i]).count('1') for i in range(n)]
    
    node_count = 0
    
    def backtrack():
        nonlocal node_count
        node_count += 1
        if time.time() - start_time > time_limit:
            return None
        
        # Find uncolored vertex with max saturation, then max degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                s = bin(sat[v]).count('1')
                if s > best_sat or (s == best_sat and degs[v] > best_deg):
                    best_sat = s
                    best_deg = degs[v]
                    best_v = v
        
        if best_v == -1:
            return True  # All colored
        
        v = best_v
        # Try colors 0, 1, 2
        for c in range(3):
            if sat[v] & (1 << c):
                continue
            # Assign color c to v
            colors[v] = c
            # Update saturation of neighbors
            av = adj_list[v]
            for u in range(n):
                if (av & (1 << u)) and colors[u] == -1:
                    sat[u] |= (1 << c)
            
            res = backtrack()
            
            if res is None:
                # Undo and propagate timeout
                colors[v] = -1
                av = adj_list[v]
                for u in range(n):
                    if (av & (1 << u)) and colors[u] == -1:
                        sat[u] &= ~(1 << c)
                return None
            
            if res:
                return True
            
            # Undo
            colors[v] = -1
            av = adj_list[v]
            for u in range(n):
                if (av & (1 << u)) and colors[u] == -1:
                    sat[u] &= ~(1 << c)
        
        return False
    
    result = backtrack()
    return result

def _greedy_maximize(adj, n, order):
    for u, v in order:
        if _can_add(adj, u, v):
            _add(adj, u, v)

def construct():
    n = 16
    start = time.time()
    deadline = start + 115.0  # 5s safety margin
    
    best_adj = None
    best_edges = 0
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Core: Grötzsch graph on vertices 0..10
    def make_grotzsch(n):
        adj = [0] * n
        # Cycle 0-1-2-3-4-0
        for i in range(5):
            _add(adj, i, (i + 1) % 5)
        # Mycielski: 5..9
        for i in range(5):
            _add(adj, i, 5 + ((i + 1) % 5))
            _add(adj, (i + 1) % 5, 5 + i)
        # Vertex 10
        for i in range(5):
            _add(adj, 5 + i, 10)
        return adj
    
    random.seed(12345)
    
    # Phase 1: Deterministic greedy from Grötzsch
    adj = make_grotzsch(n)
    _greedy_maximize(adj, n, all_pairs)
    ec = _edge_count(adj, n)
    if not _is_3_colorable(adj, n):
        best_adj = adj[:]
        best_edges = ec
    
    # Phase 2: Random restarts with greedy edge addition
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Random maximal triangle-free graph
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if _can_add(adj, u, v):
                _add(adj, u, v)
        
        ec = _edge_count(adj, n)
        if ec <= best_edges:
            continue
        
        # Check 3-colorability with short timeout
        res = _is_3_colorable(adj, n, time_limit=0.05)
        if res is None:
            continue  # Timeout, skip
        if not res:
            best_adj = adj[:]
            best_edges = ec
    
    # Phase 3: Local search - try edge removals and re-greedy
    if best_adj is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            adj = best_adj[:]
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                if adj[u] & (1 << v):
                    # Remove edge (u,v)
                    _remove(adj, u, v)
                    # Try to add other edges greedily
                    new_adj = adj[:]
                    for u2, v2 in all_pairs:
                        if _can_add(new_adj, u2, v2):
                            _add(new_adj, u2, v2)
                    ec = _edge_count(new_adj, n)
                    if ec > best_edges:
                        res = _is_3_colorable(new_adj, n, time_limit=0.1)
                        if res is not None and not res:
                            best_adj = new_adj
                            best_edges = ec
                            improved = True
    
    # Phase 4: More aggressive local search - try multiple edge removals
    if best_adj is not None and time.time() < deadline:
        for it in range(2000):
            if time.time() > deadline:
                break
            
            adj = best_adj[:]
            # Remove 1-2 random edges
            num_remove = random.randint(1, 2)
            for _ in range(num_remove):
                present = [(u, v) for u, v in all_pairs if adj[u] & (1 << v)]
                if not present:
                    break
                u, v = random.choice(present)
                _remove(adj, u, v)
            
            # Re-greedy
            for u2, v2 in all_pairs:
                if _can_add(adj, u2, v2):
                    _add(adj, u2, v2)
            
            ec = _edge_count(adj, n)
            if ec > best_edges:
                res = _is_3_colorable(adj, n, time_limit=0.1)
                if res is not None and not res:
                    best_adj = adj[:]
                    best_edges = ec
    
    # Convert best_adj to edge list
    edges = []
    if best_adj is None:
        # Fallback: return Grötzsch with isolated vertices (11 edges)
        adj = make_grotzsch(n)
        best_adj = adj
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
