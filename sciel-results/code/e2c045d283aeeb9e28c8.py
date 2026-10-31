<<<<<<< SEARCH
"""
Construct a k-regular graph with girth at least g.
Optimized search for circulant graphs C_n(1, a, b) with fallback to LUW.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    if k == 3 and g == 16:
        # Search for circulant graphs C_n(1, a, b) with girth >= 16
        result = _find_circulant_cage(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback: LUW with verified girth
        # Try smaller d first to get smaller n
        for d in [7, 9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    # General fallback: LUW with default d
    return _luw_construct(k, g)
=======
"""
Construct a k-regular graph with girth at least g.
Uses random lifts of the Petersen graph with local search for k=3, g=16.
Fallback to LUW construction.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    # For the specific case k=3, g=16:
    if k == 3 and g == 16:
        # Try random lift of Petersen graph with local search
        result = _petersen_lift_search(3, 16, deadline)
        if result is not None:
            n, edges = result
            return n, edges

        # Fallback: LUW with verified girth
        # Try smaller d first to get smaller n
        for d in [7, 9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    # General fallback: LUW with default d
    return _luw_construct(k, g)


def _petersen_lift_search(k, g, deadline):
    """
    Search for a random lift of the Petersen graph with girth >= g.
    The Petersen graph has 10 vertices, degree 3, girth 5.
    A random lift of degree m has 10*m vertices.
    We try various m values and use local search to eliminate short cycles.
    """
    # Petersen graph edges
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # For each lift degree m, the lift has 10*m vertices
    # Each vertex (i, t) where i in 0..9, t in 0..m-1
    # Edges connect (i, t) to (j, (t + w_ij) % m) for each edge (i,j) in Petersen
    
    best_result = None
    
    # Try different lift degrees
    for m in [16, 20, 24, 30, 32, 40, 48, 50, 60, 64, 80, 96, 100, 120, 128, 160, 192, 200, 240, 256, 320, 384, 400, 480, 512]:
        if time.time() > deadline - 5:
            break
        
        n = 10 * m
        
        # Try multiple random lifts
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Random weights for each Petersen edge
            weights = {}
            for (u, v) in petersen_edges:
                weights[(u, v)] = random.randint(0, m - 1)
                weights[(v, u)] = (m - weights[(u, v)]) % m if m > 1 else 0
            
            # Build the lift
            edges = []
            for (u, v) in petersen_edges:
                w = weights[(u, v)]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + ((t + w) % m)
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
            
            # Check if this lift already has girth >= g
            if _verify_girth_fast(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                break
            
            # Try local search to improve girth
            improved = _local_search_girth(n, edges, g, deadline)
            if improved is not None:
                if best_result is None or n < best_result[0]:
                    best_result = (n, improved)
                break
        
        # If we found a valid graph, check if we can try smaller m
        if best_result is not None and best_result[0] <= n:
            # We have a valid graph of size n, try to see if smaller works
            # But we've already tried smaller m values, so continue
            pass
    
    return best_result


def _local_search_girth(n, edges, g, deadline):
    """
    Local search to eliminate cycles shorter than g.
    Uses edge switching (2-switches) to modify the graph while maintaining 3-regularity.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all cycles shorter than g
    short_cycles = _find_short_cycles(n, adj, g)
    
    if not short_cycles:
        return edges
    
    # Try to eliminate short cycles via edge switching
    max_iterations = 1000
    iteration = 0
    
    while short_cycles and iteration < max_iterations:
        if time.time() > deadline - 3:
            break
        
        iteration += 1
        
        # Pick a random short cycle
        cycle = short_cycles[0]
        
        # Try to switch an edge in the cycle
        # A 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d)
        # This maintains 3-regularity if done correctly
        
        success = False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            
            # Get other neighbors of u and v
            other_u = [x for x in adj[u] if x != v]
            other_v = [x for x in adj[v] if x != u]
            
            for a in other_u:
                for b in other_v:
                    # Check if (a,b) is already an edge
                    if b in adj[a]:
                        continue
                    
                    # Perform the switch
                    # Remove (u,v) and (a,b)
                    # Add (u,a) and (v,b)
                    
                    # Check for immediate issues
                    if a == v or b == u:
                        continue
                    
                    # Tentatively apply
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].append(a)
                    adj[v].append(b)
                    adj[a].remove(u) if u in adj[a] else None
                    adj[b].remove(v) if v in adj[b] else None
                    adj[a].append(v)
                    adj[b].append(u)
                    
                    # Check if this improves girth
                    new_girth = _compute_girth(n, adj)
                    
                    if new_girth >= g or new_girth > g:
                        success = True
                        # Rebuild edges
                        new_edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    new_edges.append((i, j))
                        # Find remaining short cycles
                        short_cycles = _find_short_cycles(n, adj, g)
                        break
                    else:
                        # Revert
                        adj[u].remove(a)
                        adj[v].remove(b)
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[a].remove(v) if v in adj[a] else None
                        adj[b].remove(u) if u in adj[b] else None
                        adj[a].append(u)
                        adj[b].append(v)
            
            if success:
                break
        
        if not success:
            # Try a random 2-switch
            if time.time() > deadline - 3:
                break
            
            # Pick two random edges
            edge_list = list(edges)
            if len(edge_list) < 2:
                break
            
            e1, e2 = random.sample(range(len(edge_list)), 2)
            a, b = edge_list[e1]
            c, d = edge_list[e2]
            
            # Check if they share a vertex
            if a == c or a == d or b == c or b == d:
                continue
            
            # Check if (a,c) and (b,d) are already edges
            if c in adj[a] or d in adj[b]:
                continue
            
            # Perform switch
            adj[a].remove(b)
            adj[b].remove(a)
            adj[a].append(c)
            adj[b].append(d)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[c].append(b)
            adj[d].append(a)
            
            # Check girth
            new_girth = _compute_girth(n, adj)
            if new_girth >= g:
                new_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            new_edges.append((i, j))
                return new_edges
            else:
                # Revert
                adj[a].remove(c)
                adj[b].remove(d)
                adj[a].append(b)
                adj[b].append(a)
                adj[c].remove(b)
                adj[d].remove(a)
                adj[c].append(d)
                adj[d].append(c)
    
    return None


def _find_short_cycles(n, adj, g):
    """
    Find all cycles shorter than g using BFS from each vertex.
    Returns a list of cycles (each cycle is a list of vertices).
    """
    cycles = []
    seen_cycles = set()
    
    for src in range(n):
        # BFS from src
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        max_depth = g // 2
        
        while queue:
            u = queue.popleft()
            depth = dist[u]
            
            if depth >= max_depth:
                continue
            
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = []
                        # Path from src to u
                        cur = u
                        while cur != src:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(src)
                        cycle.reverse()
                        # Path from src to v
                        cur = v
                        while cur != src:
                            cycle.append(cur)
                            cur = parent[cur]
                        # Remove duplicate src
                        cycle = cycle[:-1]
                        
                        # Canonical form for deduplication
                        if len(cycle) > 0:
                            min_idx = cycle.index(min(cycle))
                            canonical = tuple(cycle[min_idx:] + cycle[:min_idx])
                            if canonical not in seen_cycles:
                                seen_cycles.add(canonical)
                                cycles.append(cycle)
                else:
                    dist[v] = depth + 1
                    parent[v] = u
                    queue.append(v)
    
    return cycles


def _compute_girth(n, adj):
    """
    Compute the girth of the graph (length of shortest cycle).
    """
    min_cycle = float('inf')
    
    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        queue = deque([src])
        
        while queue:
            u = queue.popleft()
            depth = dist[u]
            
            if depth >= min_cycle // 2:
                continue
            
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                else:
                    dist[v] = depth + 1
                    parent[v] = u
                    queue.append(v)
    
    return min_cycle if min_cycle != float('inf') else n
>>>>>>> REPLACE
