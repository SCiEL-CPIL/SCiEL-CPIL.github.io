import time

# ---------- Exact 3-colorability (triangle-free graph, 16 vertices) ----------
def _is_3_colorable(adj):
    """adj: list of 16 sets/lists. Returns True if graph is 3-colorable."""
    n = len(adj)
    # Precompute neighbor bitmasks for speed
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    # Order vertices by descending degree (static)
    # Using a dynamic ordering (MRV) is better but might be slower per step.
    # For n=16, static degree ordering is usually sufficient and fast.
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    # Precompute "earlier neighbors" for each vertex in the order
    # earlier_mask[i] is a bitmask of neighbors of order[i] that appear earlier in order
    earlier_mask = [0] * n
    for i, v in enumerate(order):
        mask = 0
        for w in adj[v]:
            if pos[w] < i:
                mask |= (1 << w)
        earlier_mask[i] = mask

    color = [-1] * n
    
    def backtrack(i):
        v = order[i]
        # Get colors of earlier neighbors
        # We need to know which colors are used by neighbors.
        # Instead of checking each neighbor's color, we can just check the 3 colors.
        # A color c is forbidden if any earlier neighbor has color c.
        
        # Optimization: 
        # If we maintain for each color c, a bitmask of vertices colored c, 
        # then for vertex v, color c is forbidden if (adj_mask[v] & color_mask[c]) != 0.
        # But color_mask changes dynamically.
        
        # Let's stick to the simple check but optimized.
        # used_colors = 0
        # for w in adj[v]:
        #    if pos[w] < i:
        #       if color[w] != -1:
        #          used_colors |= 1 << color[w]
        
        # Better: Iterate through the 3 colors.
        for c in range(3):
            # Check if any earlier neighbor has color c
            # We can precompute for each vertex v and each color c? No, colors are dynamic.
            
            # Simple loop over neighbors is O(deg(v)).
            # Sum of degrees is small for triangle-free (max ~ n^2/4 = 64).
            # So 16 * 64 operations is tiny.
            
            forbidden = False
            for w in adj[v]:
                if pos[w] < i and color[w] == c:
                    forbidden = True
                    break
            if not forbidden:
                color[v] = c
                if i == n - 1:
                    return True
                if backtrack(i + 1):
                    return True
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

def construct():
    start = time.time()
    deadline = start + 110.0
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
        # If we can't add an edge, try swapping an existing edge for another to allow future additions.
        # Actually, a 1-1 swap doesn't change edge count. We need to find a state where an edge CAN be added.
        # So: 
        # 1. Try to add edge.
        # 2. If fail, try 1-1 swap (remove e1, add e2) such that e2 != e1.
        # 3. Then try to add edge again.
        
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
            # We want to remove an edge e_rem and add an edge e_add (where e_add is not present)
            # such that the graph remains triangle-free and 4-chromatic.
            # This might open up space for more edges later.
            
            # To be efficient, we can limit the number of swaps or try random ones.
            # Let's try all possible removals and see if we can add a different edge.
            # This is O(E * P). E~50, P~120. 6000 checks. Each check involves _is_3_colorable.
            # This might be too slow if done exhaustively every iteration.
            # Let's do a limited random search for swaps.
            
            import random
            # Pick a random existing edge to remove
            if edges:
                idx = random.randint(0, len(edges) - 1)
                u_rem, v_rem = edges[idx]
                
                # Candidate removal
                cand_adj = [s.copy() for s in adj]
                cand_adj[u_rem].discard(v_rem)
                cand_adj[v_rem].discard(u_rem)
                cand_edges = edges[:idx] + edges[idx+1:]
                
                # Try to add a random new edge
                # We can try a few random pairs
                pairs_to_try = [(u, v) for u in range(n) for v in range(u+1, n) if v not in cand_adj[u]]
                random.shuffle(pairs_to_try)
                
                swapped = False
                for u_add, v_add in pairs_to_try[:50]: # Limit attempts
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
    for i in range(50): # Adjust based on time
        if not elapsed_ok():
            break
        
        # Use a different random seed for shuffling the pair order
        seed = i * 1000 + 42
        adj, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
            
            # Print progress to stdout if needed (optional, but good for debugging)
            # print(f"Seed {seed}: {count} edges")

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
