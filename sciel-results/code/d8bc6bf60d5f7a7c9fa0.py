import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 5 and g == 9:
        # Fano plane folding construction
        # Fano plane: 7 points {0..6}, 7 lines
        # Lines (cyclic construction): line i = {i, i+1, i+3} mod 7
        lines = []
        for i in range(7):
            lines.append((i, (i+1) % 7, (i+3) % 7))
        
        # Verify: each point on 3 lines, each line has 3 points
        # Point p is on lines where p in line
        point_lines = [[] for _ in range(7)]
        for li, line in enumerate(lines):
            for p in line:
                point_lines[p].append(li)
        
        # Antipodal pairing: point p is antipodal to line (p+1) mod 7
        # (the unique line NOT containing p)
        # Verify: point 0 is on lines 0, 2, 4. Not on lines 1, 3, 5, 6.
        # Wait, each point is on exactly 3 lines, so 4 lines don't contain it.
        # In Fano plane, each point is on 3 lines, 4 lines miss it.
        # The "antipodal" in the incidence graph (bipartite, girth 6):
        # distance 3 from point p is the set of lines at distance 3.
        # In incidence graph: p - line - point - line - point
        # Actually for Fano plane incidence graph (14 vertices, girth 6):
        # From point p, distance 1: 3 lines containing p
        # Distance 2: 6 points on those lines (other than p)
        # Distance 3: lines containing those 6 points but not p
        # Each of the 6 points is on 3 lines, one containing p, so 2 new lines each
        # But lines are shared. The 4 lines not containing p: each contains 3 points,
        # all different from p. So distance 3 = 4 lines.
        # Hmm, that's not a single antipodal vertex.
        
        # Let me reconsider. For the Fano plane, the incidence graph has girth 6.
        # The "antipodal" at distance 3 from a point is not unique.
        # 
        # Alternative: Use the known result that the 5-regular girth-9 graph on 7 vertices
        # exists. It's the complement of a perfect matching in K_7 minus... 
        # Actually, K_7 is 6-regular. Remove a 1-factor (perfect matching) to get 5-regular.
        # But K_7 has triangles! Girth 3.
        
        # Let me think differently. We need a 5-regular graph on 7 vertices with girth >= 9.
        # In a 7-vertex graph, the longest possible cycle is 7. So girth >= 9 is impossible
        # for 7 vertices! Maximum cycle length in a 7-vertex graph is 7.
        
        # So we need MORE than 7 vertices. Let me reconsider.
        
        # For girth 9, we need no cycles of length < 9. So minimum cycle is 9.
        # This requires at least 9 vertices for the cycle itself, but with degree 5,
        # we need more.
        
        # Moore bound for (5,9): 1 + 5*(1 + 4 + 4^2 + ... + 4^3) = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
        # So at least 426 vertices (actually must be even, so 428).
        
        # The folding approach: start with a (q+1)-regular bipartite graph of girth 6,
        # fold to get 2(q+1)-regular of girth 9. For q=2: 3-regular bipartite girth 6 -> 6-regular girth 9.
        # The Fano plane incidence graph is 3-regular bipartite with girth 6 (14 vertices).
        # Folding: identify antipodal vertices. But as noted, antipodal isn't unique.
        
        # Let me try a different approach: use the LUW construction but optimize it,
        # and also try random search for smaller graphs.
        
        # Actually, let me just use the LUW construction which is guaranteed to work,
        # and add a random search component for potentially smaller graphs.
        
        # LUW gives n = 2 * 5^5 = 15625 vertices. That's our baseline.
        
        # Let's try to do better with a smarter approach.
        # 
        # Key insight: The Hoffman-Singleton graph is (7,5) - 7-regular girth 5 on 50 vertices.
        # We need (5,9).
        #
        # Let me just implement the LUW construction efficiently and return it.
        # Then add a random search for smaller N if time permits.
        
        # First, try the LUW construction
        n_luw, edges_luw = _luw_construct(k, g, deadline)
        
        # Now try random search for smaller N
        # N must be even (5N must be even)
        # Moore bound ~426, so try from 428 upward
        best_n = n_luw
        best_edges = edges_luw
        
        # Try a limited random search for smaller N
        # Focus on N values where probability of existence is non-negligible
        # For (5,9), known best is 1116, so try around there and below
        
        time_left = deadline - time.time()
        if time_left > 10:
            # Try random graphs for N from 428 to min(1500, best_n - 2)
            N_start = 428
            N_end = min(1500, best_n - 2)
            
            if N_start < N_end:
                # Allocate time for random search
                search_time = time_left - 5  # keep 5s for return
                
                # Try N values in strategic order
                # Start with values near known best (1116) and work down
                N_values = list(range(N_start, N_end + 1, 2))
                
                # Prioritize: try values near 1116 first, then spread out
                # Actually, we want MINIMUM N, so go from low to high
                # But probability is very low for small N
                # Let's try a few small N, then focus on medium range
                
                tried = set()
                
                # Phase 1: Try small N with few trials
                for N in range(N_start, min(N_start + 100, N_end + 1), 2):
                    if time.time() > deadline - 5:
                        break
                    if N in tried:
                        continue
                    tried.add(N)
                    # Try 5 random graphs
                    for trial in range(5):
                        if time.time() > deadline - 5:
                            break
                        edges = _gen_random_5_reg(N)
                        if edges is None:
                            continue
                        if _check_girth_fast(edges, N, g):
                            best_n = N
                            best_edges = edges
                            return N, edges
                
                # Phase 2: Try medium N with more trials
                for N in range(max(N_start, 800), min(N_end + 1, 1200), 2):
                    if time.time() > deadline - 5:
                        break
                    if N in tried:
                        continue
                    tried.add(N)
                    # Try 20 random graphs
                    for trial in range(20):
                        if time.time() > deadline - 5:
                            break
                        edges = _gen_random_5_reg(N)
                        if edges is None:
                            continue
                        if _check_girth_fast(edges, N, g):
                            best_n = N
                            best_edges = edges
                            return N, edges
                
                # Phase 3: If time remains, try larger N
                for N in range(1200, N_end + 1, 2):
                    if time.time() > deadline - 5:
                        break
                    if N in tried:
                        continue
                    tried.add(N)
                    # Try 50 random graphs
                    for trial in range(50):
                        if time.time() > deadline - 5:
                            break
                        edges = _gen_random_5_reg(N)
                        if edges is None:
                            continue
                        if _check_girth_fast(edges, N, g):
                            best_n = N
                            best_edges = edges
                            return N, edges
        
        return best_n, best_edges
    
    else:
        return _luw_construct(k, g, deadline)

def _gen_random_5_reg(N):
    """Generate a random 5-regular graph on N vertices using configuration model."""
    if N < 6:  # Need at least 6 vertices for 5-regular (actually 2 vertices with 5 parallel edges, but we need simple)
        return None
    if N % 2 != 0:  # 5N must be even
        return None
    
    stubs = [i for i in range(N) for _ in range(5)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edges:
            return None
        edges.add(key)
    
    # Verify all vertices have degree 5
    deg = [0] * N
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 5 for d in deg):
        return None
    
    return list(edges)

def _check_girth_fast(edges, N, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex."""
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _greedy_girth_improvement(edges, N, g, deadline):
    """
    Attempts to increase the girth of a 5-regular graph by breaking short cycles.
    Returns improved edges or None if no improvement possible within time.
    """
    edge_set = set(edges)
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    current_girth = _find_shortest_cycle_len(adj, N, g)
    if current_girth >= g:
        return list(edge_set)
    
    improved = False
    last_change_time = time.time()
    
    while time.time() < deadline - 2:
        if time.time() - last_change_time > 5:
            # Stall detection: if no improvement for 5s, break
            break
            
        # Find shortest cycle
        cycle_len, cycle_edges = _find_shortest_cycle_info(adj, N, g)
        if cycle_len >= g:
            return list(edge_set)
        
        # Pick a random edge from the cycle to break
        # We want to replace (u,v) with (u,u') and (v,v') by swapping with (u',v')
        # We iterate over candidate edges (u',v')
        
        # To be efficient, we might just try all edges? No, too slow.
        # Let's try random edges first.
        
        found_swap = False
        # Try 50 random edges as the "other" edge in the 2-flip
        for _ in range(50):
            if time.time() > deadline - 2:
                return None
            if not edges: break # Should not happen
            
            # Pick random edge to swap with
            idx = random.randint(0, len(edges) - 1)
            e2 = edges[idx]
            
            # Pick random edge from cycle to break
            e1 = random.choice(cycle_edges)
            
            u, v = e1
            u2, v2 = e2
            
            # Check if e1 and e2 are disjoint
            if u in (u2, v2) or v in (u2, v2):
                continue
            
            # Proposed new edges: (u, u2) and (v, v2)
            # Or (u, v2) and (v, u2)
            # We try both if valid
            
            candidates = [
                ((u, u2), (v, v2)),
                ((u, v2), (v, u2))
            ]
            
            for new_e1, new_e2 in candidates:
                a, b = new_e1
                c, d = new_e2
                if a == b or c == d:
                    continue
                key1 = (min(a, b), max(a, b))
                key2 = (min(c, d), max(c, d))
                
                if key1 == key2:
                    continue
                if key1 in edge_set or key2 in edge_set:
                    continue
                
                # Perform swap
                edge_set.remove(e1)
                edge_set.remove(e2)
                edge_set.add(key1)
                edge_set.add(key2)
                
                # Update adjacency
                adj[u].remove(v)
                adj[v].remove(u)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[a].append(b)
                adj[b].append(a)
                adj[c].append(d)
                adj[d].append(c)
                
                # Check new girth
                new_girth = _find_shortest_cycle_len(adj, N, g)
                
                if new_girth > current_girth:
                    current_girth = new_girth
                    improved = True
                    last_change_time = time.time()
                    # Update edges list
                    edges = list(edge_set)
                    found_swap = True
                    break
                else:
                    # Revert
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    
                    edge_set.add(e1)
                    edge_set.add(e2)
                    edge_set.remove(key1)
                    edge_set.remove(key2)
            
            if found_swap:
                break
        
        if not found_swap:
            # Try more random restarts of the swap search?
            # Or break if stuck
            # Let's try a few more cycles of random edge picking
            # If still no swap, break to avoid wasting time
            # Actually, if we can't find a valid swap that improves girth, 
            # we might be in a local minimum.
            # We should break and try a new random graph.
            break

    return list(edge_set) if improved else None

def _find_shortest_cycle_len(adj, N, g):
    """
    Finds the length of the shortest cycle in the graph.
    Returns g if no cycle shorter than g is found.
    """
    best_len = g
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= best_len:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best_len:
                        best_len = cycle_len
                        if best_len == 3:
                            return 3
    return best_len

def _find_shortest_cycle_info(adj, N, g):
    """
    Finds the shortest cycle and returns its length and list of edges.
    """
    best_len = g
    best_edges = []
    
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        parent_edge = [None] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= best_len:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    parent_edge[v] = (min(u, v), max(u, v))
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best_len:
                        # Reconstruct edges
                        edges_in_cycle = set()
                        edges_in_cycle.add((min(u, v), max(u, v)))
                        
                        curr = u
                        while curr != start:
                            edges_in_cycle.add(parent_edge[curr])
                            curr = parent[curr]
                        
                        curr = v
                        while curr != start:
                            edges_in_cycle.add(parent_edge[curr])
                            curr = parent[curr]
                        
                        best_len = cycle_len
                        best_edges = list(edges_in_cycle)
                        if best_len == 3:
                            return best_len, best_edges
    return best_len, best_edges

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges

def _check_girth(adj, n, min_girth):
    """Check if graph has girth >= min_girth."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True
