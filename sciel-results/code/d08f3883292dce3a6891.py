"""Construct 91 edges on 31 vertices minimizing 4-cycles.
Uses polarity graph of PG(2,5) for a C4-free 90-edge core,
then adds one edge and performs local search (edge swaps) to minimize total C4s.
"""
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    n = 31
    q = 5
    
    # Build points of PG(2,5)
    # Each point is [x:y:z] normalized so first nonzero coord is 1
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first nonzero coordinate becomes 1
                if x != 0:
                    inv_x = pow(x, q - 2, q)  # Fermat's little theorem
                    pt = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    pt = (0, 1, (z * inv_y) % q)
                else:
                    pt = (0, 0, 1)
                if pt not in seen:
                    seen.add(pt)
                    points.append(pt)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: p=(x1,y1,z1) perp q=(x2,y2,z2) iff x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    # This is the standard polarity from the conic x^2 + y^2 = z^2
    def is_adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        return (x1 * x2 + y1 * y2 - z1 * z2) % q == 0
    
    # Build core edge list
    core_edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_adjacent(i, j):
                core_edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    # If core doesn't have exactly 90 edges, use greedy fallback
    if len(core_edges) != 90:
        return _greedy_construct()
    
    # Count C4s in core (should be 0 or very small)
    def count_c4(adj_sets):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                cn = len(adj_sets[i] & adj_sets[j])
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    core_c4s = count_c4(adj)
    
    # Find best edge to add: minimize new C4s created
    best_edge = None
    best_new_c4s = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 60:
            break
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            new_c4s = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        new_c4s += 1
            if new_c4s < best_new_c4s:
                best_new_c4s = new_c4s
                best_edge = [u, v]
    
    if best_edge is None:
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break
    
    # Add the best edge
    u, v = best_edge
    adj[u].add(v)
    adj[v].add(u)
    current_edges = core_edges + [best_edge]
    
    # Local search: swap one existing edge with a non-edge to reduce C4 count
    # A swap (remove edge e1, add edge e2) changes C4 count by:
    #   delta = C4s_after - C4s_before
    # We want delta < 0
    # 
    # Strategy: for each pair (edge_to_remove, edge_to_add), compute the change.
    # This is expensive, so we'll be smart about it.
    
    def c4_contribution_of_edge(adj_sets, u, v):
        """Count C4s that include edge (u,v)."""
        # A C4 containing (u,v) is a path of length 3 from u to v through two other vertices
        # i.e., u-a-b-v where a~u, a~b, b~v, a!=v, b!=u
        cnt = 0
        for a in adj_sets[u]:
            if a == v:
                continue
            for b in adj_sets[v]:
                if b == u:
                    continue
                if b in adj_sets[a]:
                    cnt += 1
        return cnt // 2  # each C4 counted twice (once from each direction)
    
    # Actually, let's use a more efficient approach.
    # Total C4s = sum over all pairs (i,j) of C(common_neighbors(i,j), 2) / 2
    # When we remove edge (u,v):
    #   - C4s that used (u,v) are removed
    #   - Some C4s that didn't use (u,v) may still exist (unchanged)
    #   - But removing (u,v) can also break C4s where u and v are opposite vertices
    # 
    # Let me think more carefully.
    # A C4 is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da (in some order).
    # Removing edge (u,v) removes all C4s that contain edge (u,v).
    # Adding edge (u,v) adds all C4s that contain edge (u,v) in the new graph.
    
    # For a swap: remove edge (r1,r2), add edge (a1,a2)
    # New C4 count = old C4 count - C4s_containing_(r1,r2)_in_old + C4s_containing_(a1,a2)_in_new
    # But the "new" graph after removal has different C4s for (a1,a2)
    
    # Simpler: just compute total C4s before and after each candidate swap.
    # With 91 edges, there are 91 * (465-91) = 91*374 = 34034 possible swaps.
    # Computing C4 count from scratch is O(n^2 * avg_degree^2) which is manageable.
    
    def compute_total_c4(adj_sets):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = adj_sets[i] & adj_sets[j]
                cn = len(common)
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    current_c4s = compute_total_c4(adj)
    
    # Build edge set for quick lookup
    edge_set = set()
    for e in current_edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Local search: try all swaps
    best_swap = None
    best_c4s = current_c4s
    
    # Precompute list of edges as tuples
    edge_list = [(min(e[0], e[1]), max(e[0], e[1])) for e in current_edges]
    
    # All non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
    
    # Try swaps: for each edge to remove, for each non-edge to add
    # To keep it fast, we'll do a limited search
    max_swaps_to_try = 20000
    
    swaps_tried = 0
    for r_e in edge_list:
        if time.time() > deadline - 10:
            break
        r_u, r_v = r_e
        for a_e in non_edges:
            if swaps_tried >= max_swaps_to_try:
                break
            a_u, a_v = a_e
            if a_e == r_e:
                continue
            swaps_tried += 1
            
            # Try the swap: remove r_e, add a_e
            adj[r_u].discard(r_v)
            adj[r_v].discard(r_u)
            adj[a_u].add(a_v)
            adj[a_v].add(a_u)
            
            new_c4s = compute_total_c4(adj)
            
            # Undo
            adj[r_u].add(r_v)
            adj[r_v].add(r_u)
            adj[a_u].discard(a_v)
            adj[a_v].discard(a_u)
            
            if new_c4s < best_c4s:
                best_c4s = new_c4s
                best_swap = (r_e, a_e)
                if new_c4s == 0:
                    break
        if time.time() > deadline - 10:
            break
        if best_c4s == 0:
            break
    
    # Apply the best swap if found
    if best_swap is not None:
        r_e, a_e = best_swap
        r_u, r_v = r_e
        a_u, a_v = a_e
        adj[r_u].discard(r_v)
        adj[r_v].discard(r_u)
        adj[a_u].add(a_v)
        adj[a_v].add(a_u)
        
        # Update edge list
        current_edges = [e for e in current_edges 
                        if (min(e[0],e[1]), max(e[0],e[1])) != r_e]
        current_edges.append([a_u, a_v])
    
    # Second pass of local search if time permits
    if time.time() < deadline - 15:
        edge_set = set()
        for e in current_edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        edge_list = [(min(e[0], e[1]), max(e[0], e[1])) for e in current_edges]
        non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
        
        best_swap2 = None
        best_c4s2 = compute_total_c4(adj)
        
        swaps_tried2 = 0
        for r_e in edge_list:
            if time.time() > deadline - 10:
                break
            r_u, r_v = r_e
            for a_e in non_edges:
                if swaps_tried2 >= max_swaps_to_try:
                    break
                a_u, a_v = a_e
                if a_e == r_e:
                    continue
                swaps_tried2 += 1
                
                adj[r_u].discard(r_v)
                adj[r_v].discard(r_u)
                adj[a_u].add(a_v)
                adj[a_v].add(a_u)
                
                new_c4s = compute_total_c4(adj)
                
                adj[r_u].add(r_v)
                adj[r_v].add(r_u)
                adj[a_u].discard(a_v)
                adj[a_v].discard(a_u)
                
                if new_c4s < best_c4s2:
                    best_c4s2 = new_c4s
                    best_swap2 = (r_e, a_e)
                    if new_c4s == 0:
                        break
            if time.time() > deadline - 10:
                break
            if best_c4s2 == 0:
                break
        
        if best_swap2 is not None:
            r_e, a_e = best_swap2
            r_u, r_v = r_e
            a_u, a_v = a_e
            adj[r_u].discard(r_v)
            adj[r_v].discard(r_u)
            adj[a_u].add(a_v)
            adj[a_v].add(a_u)
            
            current_edges = [e for e in current_edges 
                            if (min(e[0],e[1]), max(e[0],e[1])) != r_e]
            current_edges.append([a_u, a_v])
    
    # Verify
    assert len(current_edges) == 91
    final_edge_set = set(tuple(sorted(e)) for e in current_edges)
    assert len(final_edge_set) == 91
    
    return current_edges

def _greedy_construct():
    """Greedy construction: add edges one by one, each time choosing the edge
    that adds the fewest new C4s."""
    import random
    random.seed(42)
    
    t0 = time.time()
    dl = t0 + 105.0
    n = 31
    target = 91
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    def count_new_c4s(u, v):
        """Count C4s created by adding edge (u,v)."""
        cnt = 0
        for a in adj[u]:
            for b in adj[v]:
                if b in adj[a]:
                    cnt += 1
        return cnt
    
    remaining = list(all_edges)
    
    for _ in range(target):
        if time.time() > dl:
            break
        best_e = None
        best_cnt = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            cnt = count_new_c4s(u, v)
            if cnt < best_cnt:
                best_cnt = cnt
                best_e = e
            if cnt == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    # If we didn't reach target, add arbitrary remaining edges
    while len(edges) < target and remaining:
        e = remaining.pop()
        u, v = e
        if v not in adj[u]:
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
    
    return edges[:target]
