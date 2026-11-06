import time

# ---------- Exact 3-colorability (triangle-free graph, 16 vertices) ----------
def _is_3_colorable(adj):
    """adj: list of 16 sets/lists. Returns True if graph is 3-colorable."""
    n = len(adj)
    # Precompute neighbor bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    # Order vertices by descending degree (static)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    color = [-1] * n
    color_mask = [0, 0, 0]  # bitmask of vertices with color c
    
    def backtrack(i):
        v = order[i]
        # Check each color
        for c in range(3):
            # Color c is forbidden if any neighbor of v has color c
            if adj_mask[v] & color_mask[c]:
                continue
            # Assign color c to v
            color[v] = c
            color_mask[c] |= (1 << v)
            if i == n - 1:
                return True
            if backtrack(i + 1):
                return True
            # Undo
            color_mask[c] &= ~(1 << v)
        color[v] = -1
        return False

    return backtrack(0)

def _build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    return True

def _can_add_edge(u, v, adj):
    if u == v or v in adj[u]:
        return False
    # Check triangle-free: no common neighbor
    au = adj[u]
    for w in au:
        if w in adj[v]:
            return False
    return True

def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def _initial_4chromatic_graph():
    """
    Start with a known 4-chromatic triangle-free graph on 11 vertices (Mycielski of C5 = Grötzsch),
    then extend to 16 vertices with some structure to allow denser packing.
    """
    # Grötzsch graph on vertices 0-10
    edges = _groetzsch_edges()
    adj = _build_adj(edges, 16)
    
    # Add vertices 11-15 with some edges to make it harder to 3-color
    # Connect them in a way that preserves triangle-free but adds constraints
    # For now, just return the base and let greedy fill the rest
    return adj, edges

def construct():
    start = time.time()
    deadline = start + 115.0  # Leave 5s safety margin for 120s limit
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline

    def try_add_edge(u, v, adj, edges):
        if not _can_add_edge(u, v, adj):
            return None
        new_adj = [s.copy() for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if _is_3_colorable(new_adj):
            return None
        return new_adj, edges + [[u, v]]

    def run_greedy(seed_order=None):
        # Start with Groetzsch
        base_edges = _groetzsch_edges()
        adj = _build_adj(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free(adj) or _is_3_colorable(adj):
            return adj, edges, len(edges)

        # Determine order of pairs to try
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        if seed_order is not None:
            import random
            random.seed(seed_order)
            random.shuffle(all_pairs)
        else:
            # Sort by potential gain? Or just fixed order.
            pass

        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    changed = True
        
        # Local Search: 1-1 Swaps to escape local optima
        improved = True
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Try to add any edge
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    improved = True
                    break # Restart scan
            
            if improved or not elapsed_ok():
                continue

            # Phase 2: Try 1-1 swap to change structure
            import random
            if edges:
                idx = random.randint(0, len(edges) - 1)
                u_rem, v_rem = edges[idx]
                
                # Candidate removal
                cand_adj = [s.copy() for s in adj]
                cand_adj[u_rem].discard(v_rem)
                cand_adj[v_rem].discard(u_rem)
                cand_edges = edges[:idx] + edges[idx+1:]
                
                # Try to add a random new edge
                pairs_to_try = [(u, v) for u in range(n) for v in range(u+1, n) if v not in cand_adj[u]]
                random.shuffle(pairs_to_try)
                
                swapped = False
                for u_add, v_add in pairs_to_try[:80]: # Increased from 50 to 80
                    if not elapsed_ok():
                        break
                    if not _can_add_edge(u_add, v_add, cand_adj):
                        continue
                    tmp_adj = [s.copy() for s in cand_adj]
                    tmp_adj[u_add].add(v_add)
                    tmp_adj[v_add].add(u_add)
                    if _is_3_colorable(tmp_adj):
                        continue
                    
                    # Valid swap
                    adj = tmp_adj
                    edges = cand_edges + [[u_add, v_add]]
                    swapped = True
                    break
                
                if swapped:
                    improved = True
        
        return adj, edges, len(edges)

    # Run multiple greedy+local search instances with different random seeds
    import random
    max_restarts = 100  # Increased from 50, but deadline guard will stop early
    for i in range(max_restarts):
        if not elapsed_ok():
            break
        
        # Use a different random seed for shuffling the pair order
        seed = i * 1000 + 42
        adj, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]

    # Final validation
    if best_edges:
        final_adj = _build_adj(best_edges, n)
        assert _is_triangle_free(final_adj)
        assert not _is_3_colorable(final_adj)
        
        return best_edges
    else:
        # Fallback to Groetzsch if something went wrong
        base_edges = _groetzsch_edges()
        return [list(e) for e in base_edges]
