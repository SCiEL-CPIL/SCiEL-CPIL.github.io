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
        
    # Precompute "earlier neighbors" for each vertex in the order
    earlier_mask = [0] * n
    for i, v in enumerate(order):
        mask = 0
        for w in adj[v]:
            if pos[w] < i:
                mask |= (1 << w)
        earlier_mask[i] = mask

    # Use bitmask for colors: color_mask[c] = bitmask of vertices with color c
    color = [-1] * n
    color_mask = [0, 0, 0]
    
    def backtrack(i):
        v = order[i]
        em = earlier_mask[i]
        
        # Find available colors using bitmask operations
        for c in range(3):
            # Color c is forbidden if any earlier neighbor has color c
            if (adj_mask[v] & color_mask[c]) == 0:
                color[v] = c
                color_mask[c] |= (1 << v)
                if i == n - 1:
                    return True
                if backtrack(i + 1):
                    return True
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
    au = adj[u]
    for w in au:
        if w in adj[v]:
            return False
    return True

def _edge_count(adj, n=16):
    return sum(len(adj[u]) for u in range(n)) // 2

def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def _initial_4chromatic_graph(n=16):
    """Build a 4-chromatic triangle-free graph on n vertices.
    Uses the Mycielski construction starting from C5."""
    if n == 5:
        return [(i, (i + 1) % 5) for i in range(5)]
    if n == 11:
        return _groetzsch_edges()
    
    # For n=16, start with Mycielski of C5 (11 vertices) and add 5 more vertices
    # Connect new vertices to create additional edges while maintaining triangle-free and 4-chromatic
    edges = _groetzsch_edges()
    adj = _build_adj(edges, n)
    
    # Add vertices 11-15. Connect them strategically to increase edge count
    # Each new vertex can connect to non-adjacent vertices from existing graph
    for new_v in range(11, n):
        # Find vertices that are not adjacent to each other (to avoid triangles)
        # Connect to as many as possible
        candidates = list(range(11))
        # Shuffle for variety
        import random
        random.shuffle(candidates)
        for v in candidates:
            if len(adj[new_v]) >= 5:  # Limit degree to keep it manageable
                break
            if _can_add_edge(new_v, v, adj):
                adj[new_v].add(v)
                adj[v].add(new_v)
                edges.append([new_v, v])
    
    # Verify it's still 4-chromatic
    if _is_3_colorable(adj):
        # Fallback: use a different construction
        # Try connecting new vertices in a cycle pattern
        adj2 = _build_adj(_groetzsch_edges(), n)
        edges2 = [list(e) for e in _groetzsch_edges()]
        for i, new_v in enumerate(range(11, n)):
            # Connect to a few vertices to form a structure
            for target in [0, 1, 2]:
                if _can_add_edge(new_v, target, adj2):
                    adj2[new_v].add(target)
                    adj2[target].add(new_v)
                    edges2.append([new_v, target])
        if not _is_3_colorable(adj2):
            return edges2
    
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

    def greedy_add_edges(adj, edges, seed_order=None):
        """Greedy phase: add edges in a smart order."""
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        if seed_order is not None:
            import random
            rng = random.Random(seed_order)
            rng.shuffle(all_pairs)
        
        # Sort by a heuristic: prefer edges that connect high-degree vertices
        # But also consider which edges are "critical" for 3-colorability
        def edge_heuristic(u, v):
            # Score: prefer adding edges between vertices with higher degrees
            return len(adj[u]) + len(adj[v])
        
        if seed_order is None:
            all_pairs.sort(key=lambda p: -edge_heuristic(p[0], p[1]))
        
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
                    # Re-sort remaining pairs based on new degrees
                    if seed_order is None:
                        remaining = [(a, b) for a, b in all_pairs if b not in adj[a]]
                        remaining.sort(key=lambda p: -(len(adj[p[0]]) + len(adj[p[1]])))
                        all_pairs = remaining
        
        return adj, edges

    def local_search(adj, edges, seed_order=None):
        """Local search: try 1-1 and 2-2 swaps to escape local optima."""
        import random
        rng = random.Random(seed_order)
        
        improved = True
        swap_count = 0
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Try to add any edge
            all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
            rng.shuffle(all_pairs)
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    improved = True
                    break
            
            if improved or not elapsed_ok():
                continue

            # Phase 2: Try 1-1 swaps
            # Remove one edge, add another (different) edge
            edge_list = [(u, v) for u, v in edges]
            rng.shuffle(edge_list)
            
            for u_rem, v_rem in edge_list[:20]:  # Limit attempts
                if not elapsed_ok():
                    break
                
                cand_adj = [s.copy() for s in adj]
                cand_adj[u_rem].discard(v_rem)
                cand_adj[v_rem].discard(u_rem)
                cand_edges = [e for e in edges if not (e[0] == u_rem and e[1] == v_rem)]
                
                # Try to add a new edge
                new_pairs = [(u, v) for u in range(n) for v in range(u+1, n) 
                            if v not in cand_adj[u] and not (u == u_rem and v == v_rem) 
                            and not (u == v_rem and v == u_rem)]
                rng.shuffle(new_pairs)
                
                for u_add, v_add in new_pairs[:30]:
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
                    improved = True
                    swap_count += 1
                    break
                
                if improved:
                    break
            
            # Phase 3: Try 2-2 swaps (remove 2 edges, add 2 edges)
            if not improved and len(edges) >= 4:
                edge_list2 = [(u, v) for u, v in edges]
                rng.shuffle(edge_list2)
                
                for i in range(min(10, len(edge_list2))):
                    if not elapsed_ok():
                        break
                    u1_rem, v1_rem = edge_list2[i]
                    for j in range(i+1, min(i+10, len(edge_list2))):
                        u2_rem, v2_rem = edge_list2[j]
                        
                        cand_adj = [s.copy() for s in adj]
                        cand_adj[u1_rem].discard(v1_rem)
                        cand_adj[v1_rem].discard(u1_rem)
                        cand_adj[u2_rem].discard(v2_rem)
                        cand_adj[v2_rem].discard(u2_rem)
                        cand_edges = [e for e in edges if e not in [[u1_rem, v1_rem], [u2_rem, v2_rem]]]
                        
                        # Try to add 2 new edges
                        new_pairs = [(u, v) for u in range(n) for v in range(u+1, n) if v not in cand_adj[u]]
                        rng.shuffle(new_pairs)
                        
                        added = []
                        for u_add, v_add in new_pairs[:20]:
                            if len(added) == 2:
                                break
                            if not _can_add_edge(u_add, v_add, cand_adj):
                                continue
                            tmp_adj = [s.copy() for s in cand_adj]
                            tmp_adj[u_add].add(v_add)
                            tmp_adj[v_add].add(u_add)
                            if _is_3_colorable(tmp_adj):
                                continue
                            cand_adj = tmp_adj
                            added.append([u_add, v_add])
                        
                        if len(added) == 2:
                            adj = cand_adj
                            edges = cand_edges + added
                            improved = True
                            swap_count += 1
                            break
                    if improved:
                        break
        
        return adj, edges, swap_count

    def run_search(seed_order=None):
        """Run a complete search instance."""
        # Start with initial 4-chromatic graph
        base_edges = _initial_4chromatic_graph(n)
        adj = _build_adj(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free(adj) or _is_3_colorable(adj):
            return adj, edges, len(edges)

        # Greedy phase
        adj, edges = greedy_add_edges(adj, edges, seed_order)
        
        # Local search phase
        adj, edges, swaps = local_search(adj, edges, seed_order)
        
        return adj, edges, len(edges)

    # Run multiple search instances with different random seeds
    import random
    for i in range(80):  # More instances due to faster color check
        if not elapsed_ok():
            break
        
        seed = i * 7919 + 12345
        adj, edges, count = run_search(seed)
        
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
