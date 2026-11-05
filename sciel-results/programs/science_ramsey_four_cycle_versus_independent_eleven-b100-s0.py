import time
import random
import sys
import numpy as np

def _construct_polarity_graph(q):
    """Construct the polarity graph of PG(2, q). C4-free."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [0] * n_pts  # bitmask adjacency
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    edges = []
    for i in range(n_pts):
        m = adj[i]
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
    
    return n_pts, edges, adj


def _has_independent_set(adj, n, limit):
    """Check if graph has independent set of size >= limit. Uses branch and bound with bitmasks."""
    if n <= 64:
        all_mask = (1 << n) - 1
        
        # Precompute non-neighbor masks (for independent set, we need vertices not adjacent to chosen ones)
        # When we pick vertex v, the remaining candidates must be non-neighbors of v (excluding v itself)
        
        # Use iterative deepening / branch and bound
        # State: available_mask (vertices that can still be added to independent set)
        # We want to know if we can pick 'limit' vertices from available_mask such that no two are adjacent
        
        def _search(avail, depth_needed):
            if depth_needed <= 0:
                return True
            if avail == 0:
                return False
            if avail.bit_count() < depth_needed:
                return False
            
            # Pick a vertex with high degree in the available subgraph (to prune more)
            # Actually for independent set, we want to pick vertices that are "central" 
            # so excluding them removes many options
            best_v = -1
            best_deg = -1
            m = avail
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                # Degree of v in available subgraph
                deg = (adj[v] & avail).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                return False
            
            v_bit = 1 << best_v
            
            # Option 1: Include best_v in independent set
            # Then we cannot include any neighbor of best_v
            new_avail = avail & ~(adj[best_v] | v_bit)
            if _search(new_avail, depth_needed - 1):
                return True
            
            # Option 2: Exclude best_v
            new_avail = avail & ~v_bit
            if _search(new_avail, depth_needed):
                return True
            
            return False
        
        return _search(all_mask, limit)
    else:
        # Fallback for n > 64 (unlikely given n <= 43)
        adj_set = [set() for _ in range(n)]
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                adj_set[i].add(j)
                m ^= lsb
        
        def _search(avail, depth_needed):
            if depth_needed <= 0:
                return True
            if not avail:
                return False
            if len(avail) < depth_needed:
                return False
            
            best_v = -1
            best_deg = -1
            for v in avail:
                deg = len(adj_set[v] & avail)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Include
            new_avail = avail - adj_set[best_v] - {best_v}
            if _search(new_avail, depth_needed - 1):
                return True
            
            # Exclude
            new_avail = avail - {best_v}
            if _search(new_avail, depth_needed):
                return True
            
            return False
        
        return _search(set(range(n)), limit)


def _is_c4_free(adj, n):
    """Check C4-free: no two vertices have >= 2 common neighbors."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # has at least 2 bits set
                return False
    return True


def _try_add_vertex(adj, n, limit=11, deadline=None):
    """
    Try to add a new vertex to the graph. Returns (new_adj, new_n, edges_to_add) or (None, None, None).
    The new vertex must have neighbors S such that:
    1. For all u, v in S: adj[u] & adj[v] == 0 (C4-free condition)
    2. The resulting graph has no independent set of size >= limit
    """
    # Find a maximal set S of vertices with pairwise disjoint neighborhoods
    # This is equivalent to finding a clique in the "conflict" graph where
    # vertices u,v are connected if adj[u] & adj[v] != 0
    
    # Greedy: try random orderings, build S greedily
    candidates = list(range(n))
    
    best_S = set()
    best_result = None
    
    for attempt in range(200):
        if deadline and time.time() > deadline - 5:
            break
        
        random.shuffle(candidates)
        S = set()
        for v in candidates:
            valid = True
            for u in S:
                if adj[u] & adj[v]:
                    valid = False
                    break
            if valid:
                S.add(v)
        
        if len(S) <= len(best_S):
            continue
        best_S = S.copy()
        
        # Try adding vertex with neighbors S
        new_n = n + 1
        new_adj = adj[:] + [0]
        new_adj[n] = 0
        for u in S:
            new_adj[n] |= (1 << u)
            new_adj[u] |= (1 << n)
        
        # Check C4-free for new vertex
        c4_ok = True
        for u in S:
            # u and new vertex share no neighbors (by construction, S has disjoint neighborhoods)
            # But we need to check: do any two vertices in S have common neighbors with new vertex?
            # New vertex's neighbors are S. For u, v in S, common neighbors of u and new vertex = N(u) ∩ S
            # For C4: no two vertices can have 2+ common neighbors
            # u and new vertex: common neighbors = N(u) ∩ S. Since S has pairwise disjoint neighborhoods,
            # N(u) ∩ S can only contain u itself if u is in S... wait, u is in S, and N(u) ∩ S:
            # if v in S and v in N(u), then adj[u] & adj[v] != 0, contradicting our construction.
            # So N(u) ∩ S = empty (since no two vertices in S are adjacent to each other in terms of sharing neighbors)
            # Actually wait: S vertices have pairwise disjoint neighborhoods, meaning for u,v in S, N(u) ∩ N(v) = empty.
            # But u and v themselves: is v in N(u)? Not necessarily. The condition is about common NEIGHBORS.
            # For u and new_vertex: common neighbors = N(u) ∩ N(new_vertex) = N(u) ∩ S
            # For this to be < 2, we need |N(u) ∩ S| <= 1
            # Since S has pairwise disjoint neighborhoods, if v in S and v in N(u), then for any w in S, w != v:
            #   w in N(u) would mean u and w share neighbor v, so N(u) ∩ N(w) contains v, contradiction.
            # So |N(u) ∩ S| <= 1. Good.
            # For new_vertex and any existing vertex w not in S: N(new_vertex) ∩ N(w) = S ∩ N(w)
            # This could be large! We need |S ∩ N(w)| <= 1 for all w.
            # This is automatically satisfied if S has pairwise disjoint neighborhoods? No.
            # Actually: for w not in S, S ∩ N(w) = {v in S : v in N(w)}
            # If v1, v2 in S ∩ N(w), then v1 and v2 both adjacent to w, so w in N(v1) ∩ N(v2),
            # contradicting that S has pairwise disjoint neighborhoods.
            # So |S ∩ N(w)| <= 1 for all w. Good.
            # So C4-free is maintained.
        
            # Check independent set
            if not _has_independent_set(new_adj, new_n, limit):
                edges_to_add = [[n, u] for u in S]
                return new_adj, new_n, edges_to_add
    
    # If we didn't find a valid S with the best greedy, try the best_S found
    if best_S:
        new_n = n + 1
        new_adj = adj[:] + [0]
        new_adj[n] = 0
        for u in best_S:
            new_adj[n] |= (1 << u)
            new_adj[u] |= (1 << n)
        
        if not _has_independent_set(new_adj, new_n, limit):
            edges_to_add = [[n, u] for u in best_S]
            return new_adj, new_n, edges_to_add
    
    return None, None, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_n = 0
    best_edges = []
    
    # Start with q=5 polarity graph (31 vertices)
    n, edges, adj = _construct_polarity_graph(5)
    
    if not _is_c4_free(adj, n):
        return {'n': 0, 'edges': []}
    
    if _has_independent_set(adj, n, 11):
        return {'n': 0, 'edges': []}
    
    best_n = n
    best_edges = [e[:] for e in edges]
    
    # Try to add vertices
    current_adj = adj[:]
    current_n = n
    current_edges = [e[:] for e in edges]
    
    while time.time() < deadline - 25:
        result = _try_add_vertex(current_adj, current_n, limit=11, deadline=deadline)
        if result[0] is not None:
            current_adj, current_n, new_edges = result
            current_edges.extend(new_edges)
            if current_n > best_n:
                best_n = current_n
                best_edges = [e[:] for e in current_edges]
        else:
            # Try a few more random restarts before giving up
            found = False
            for _ in range(50):
                if time.time() > deadline - 15:
                    break
                result = _try_add_vertex(current_adj, current_n, limit=11, deadline=deadline)
                if result[0] is not None:
                    current_adj, current_n, new_edges = result
                    current_edges.extend(new_edges)
                    if current_n > best_n:
                        best_n = current_n
                        best_edges = [e[:] for e in current_edges]
                    found = True
                    break
            if not found:
                break
    
    return {'n': best_n, 'edges': best_edges}
