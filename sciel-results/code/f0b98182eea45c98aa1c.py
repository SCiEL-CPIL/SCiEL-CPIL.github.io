import time

# ---------- Exact 3-colorability (triangle-free graph, 16 vertices) ----------
def _is_3_colorable(adj_mask):
    """
    adj_mask: list of 16 integers, where bit v is set if u-v is an edge.
    Returns True if graph is 3-colorable.
    """
    n = len(adj_mask)
    
    color = [-1] * n
    color_mask = [0, 0, 0]
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    # Precompute a list of vertices sorted by degree descending for tie-breaking
    # Actually, dynamic selection is better. We'll just iterate.
    
    def backtrack(uncolored_count):
        if uncolored_count == 0:
            return True
        
        best_v = -1
        best_num_avail = 4
        best_deg = -1
        best_mask = 0
        
        for v in range(n):
            if color[v] != -1:
                continue
            
            # Check available colors using bitmasks
            # avail bit c is set if no neighbor has color c
            avail = 0
            if not (adj_mask[v] & color_mask[0]):
                avail |= 1
            if not (adj_mask[v] & color_mask[1]):
                avail |= 2
            if not (adj_mask[v] & color_mask[2]):
                avail |= 4
                
            if avail == 0:
                return False
            
            num_avail = bin(avail).count('1')
            
            # Most constrained first (fewest available colors), then highest degree
            if num_avail < best_num_avail or (num_avail == best_num_avail and degrees[v] > best_deg):
                best_v = v
                best_num_avail = num_avail
                best_deg = degrees[v]
                best_mask = avail
        
        if best_v == -1:
            return True
            
        v = best_v
        for c in range(3):
            if not (best_mask & (1 << c)):
                continue
            
            color[v] = c
            color_mask[c] |= (1 << v)
            
            if backtrack(uncolored_count - 1):
                return True
            
            color_mask[c] &= ~(1 << v)
            color[v] = -1
            
        return False

    return backtrack(n)

def _build_adj_mask(edges, n=16):
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _is_triangle_free_mask(adj_mask):
    n = len(adj_mask)
    for u in range(n):
        # For each neighbor v of u, check if u and v share any common neighbor
        # Common neighbors of u and v are adj_mask[u] & adj_mask[v]
        # We need to ensure that for all v in adj_mask[u], (adj_mask[u] & adj_mask[v]) == 0
        # Wait, if v is a neighbor of u, then v is in adj_mask[u].
        # If there is a w in adj_mask[u] & adj_mask[v], then u-w and v-w are edges, and u-v is an edge -> triangle.
        # So we just need to check if for any v in adj_mask[u], adj_mask[u] & adj_mask[v] is non-zero.
        # Note: v is in adj_mask[u]. If w=v, v is in adj_mask[u] and adj_mask[v]? No, no self loops.
        # So we just iterate through bits of adj_mask[u].
        m = adj_mask[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj_mask[u] & adj_mask[v]:
                return False
            m -= lsb
    return True

def _can_add_edge_mask(u, v, adj_mask):
    if u == v:
        return False
    if adj_mask[u] & (1 << v):
        return False
    # Check triangle-free: no common neighbor
    # Common neighbors of u and v are adj_mask[u] & adj_mask[v]
    if adj_mask[u] & adj_mask[v]:
        return False
    return True

def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def construct():
    start = time.time()
    deadline = start + 115.0  # Leave 5s safety margin for 120s limit
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline

    def run_greedy(seed_order=None):
        # Start with Groetzsch
        base_edges = _groetzsch_edges()
        adj_mask = _build_adj_mask(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free_mask(adj_mask) or _is_3_colorable(adj_mask):
            return adj_mask, edges, len(edges)

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        if seed_order is not None:
            import random
            rng = random.Random(seed_order)
            rng.shuffle(all_pairs)
        else:
            all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

        # Greedy addition phase
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if adj_mask[u] & (1 << v):
                    continue
                if _can_add_edge_mask(u, v, adj_mask):
                    # Try adding
                    new_adj_mask = adj_mask[:]
                    new_adj_mask[u] |= (1 << v)
                    new_adj_mask[v] |= (1 << u)
                    if not _is_3_colorable(new_adj_mask):
                        adj_mask = new_adj_mask
                        edges.append([u, v])
                        changed = True
        
        # Local Search
        import random
        rng = random.Random(seed_order if seed_order is not None else 0)
        
        improved = True
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Try to add any edge (systematic scan)
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if adj_mask[u] & (1 << v):
                    continue
                if _can_add_edge_mask(u, v, adj_mask):
                    new_adj_mask = adj_mask[:]
                    new_adj_mask[u] |= (1 << v)
                    new_adj_mask[v] |= (1 << u)
                    if not _is_3_colorable(new_adj_mask):
                        adj_mask = new_adj_mask
                        edges.append([u, v])
                        improved = True
                        break
            
            if improved or not elapsed_ok():
                continue

            # Phase 2: Try 1-1 swaps
            swapped = False
            if edges:
                for _ in range(20):
                    if not elapsed_ok():
                        break
                    idx = rng.randint(0, len(edges) - 1)
                    u_rem, v_rem = edges[idx]
                    
                    cand_adj_mask = adj_mask[:]
                    cand_adj_mask[u_rem] &= ~(1 << v_rem)
                    cand_adj_mask[v_rem] &= ~(1 << u_rem)
                    cand_edges = edges[:idx] + edges[idx+1:]
                    
                    pairs_to_try = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if not (cand_adj_mask[u] & (1 << v)) and _can_add_edge_mask(u, v, cand_adj_mask):
                                pairs_to_try.append((u, v))
                    
                    if not pairs_to_try:
                        continue
                    
                    rng.shuffle(pairs_to_try)
                    
                    for u_add, v_add in pairs_to_try[:30]:
                        if not elapsed_ok():
                            break
                        tmp_adj_mask = cand_adj_mask[:]
                        tmp_adj_mask[u_add] |= (1 << v_add)
                        tmp_adj_mask[v_add] |= (1 << u_add)
                        if not _is_3_colorable(tmp_adj_mask):
                            adj_mask = tmp_adj_mask
                            edges = cand_edges + [[u_add, v_add]]
                            swapped = True
                            break
                    
                    if swapped:
                        break
            
            if swapped:
                improved = True
            
            # Phase 3: Shake
            if not improved and elapsed_ok():
                if len(edges) > 5:
                    shake_size = rng.randint(2, 3)
                    shake_indices = rng.sample(range(len(edges)), min(shake_size, len(edges)))
                    shake_set = set(shake_indices)
                    
                    shake_adj_mask = adj_mask[:]
                    shake_edges = []
                    for i, (u, v) in enumerate(edges):
                        if i not in shake_set:
                            shake_edges.append([u, v])
                        else:
                            shake_adj_mask[u] &= ~(1 << v)
                            shake_adj_mask[v] &= ~(1 << u)
                    
                    for u, v in all_pairs:
                        if not elapsed_ok():
                            break
                        if shake_adj_mask[u] & (1 << v):
                            continue
                        if not _can_add_edge_mask(u, v, shake_adj_mask):
                            continue
                        tmp_adj_mask = shake_adj_mask[:]
                        tmp_adj_mask[u] |= (1 << v)
                        tmp_adj_mask[v] |= (1 << u)
                        if not _is_3_colorable(tmp_adj_mask):
                            shake_adj_mask = tmp_adj_mask
                            shake_edges.append([u, v])
                    
                    if len(shake_edges) >= len(edges):
                        adj_mask = shake_adj_mask
                        edges = shake_edges
                        improved = True
        
        return adj_mask, edges, len(edges)

    max_restarts = 200
    for i in range(max_restarts):
        if not elapsed_ok():
            break
        
        seed = i * 7919 + 13
        adj_mask, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
            
        if best_count >= 45 and time.time() > deadline - 30:
            break

    if best_edges:
        final_adj_mask = _build_adj_mask(best_edges, n)
        if not _is_triangle_free_mask(final_adj_mask) or _is_3_colorable(final_adj_mask):
            base_edges = _groetzsch_edges()
            return [list(e) for e in base_edges]
        return best_edges
    else:
        base_edges = _groetzsch_edges()
        return [list(e) for e in base_edges]
