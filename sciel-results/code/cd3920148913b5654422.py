import time
import random

def _is_3_colorable(adj_mask, n=16):
    """Fast 3-colorability check using bitmasks. adj_mask: list of 16 ints."""
    # Order by degree descending
    order = sorted(range(n), key=lambda v: -adj_mask[v].bit_count())
    color_mask = [0, 0, 0]
    
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        for c in range(3):
            if adj_mask[v] & color_mask[c]:
                continue
            color_mask[c] |= (1 << v)
            if backtrack(i + 1):
                return True
            color_mask[c] &= ~(1 << v)
        return False
    
    return backtrack(0)

def _build_adj_mask(edges, n=16):
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _is_triangle_free(adj_mask, n=16):
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                if adj_mask[u] & adj_mask[v]:
                    return False
    return True

def _can_add_edge(u, v, adj_mask):
    if u == v or (adj_mask[u] & (1 << v)):
        return False
    if adj_mask[u] & adj_mask[v]:
        return False
    return True

def _groetzsch_edges():
    """Groetzsch graph: 11 vertices, triangle-free, 4-chromatic."""
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        edges.append([i, 5 + i])
        edges.append([(i + 1) % 5, 5 + i])
    return edges

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin for 150s limit
    n = 16
    
    def elapsed_ok():
        return time.time() < deadline
    
    best_edges = []
    best_count = -1
    
    def try_add_edge(u, v, adj_mask, edges):
        if not _can_add_edge(u, v, adj_mask):
            return None
        new_mask = adj_mask[:]
        new_mask[u] |= (1 << v)
        new_mask[v] |= (1 << u)
        if _is_3_colorable(new_mask, n):
            return None
        return new_mask, edges + [[u, v]]
    
    def greedy_fill(adj_mask, edges, pair_order):
        """Greedy: try to add edges in given order."""
        for u, v in pair_order:
            if not elapsed_ok():
                break
            if (adj_mask[u] & (1 << v)):
                continue
            res = try_add_edge(u, v, adj_mask, edges)
            if res is not None:
                adj_mask, edges = res
        return adj_mask, edges
    
    def local_search(adj_mask, edges, max_time_frac=0.3):
        """Local search: 1-add and 1-1 swap."""
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        improved = True
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Try to add any edge
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if (adj_mask[u] & (1 << v)):
                    continue
                res = try_add_edge(u, v, adj_mask, edges)
                if res is not None:
                    adj_mask, edges = res
                    improved = True
                    break
            
            if improved or not elapsed_ok():
                continue
            
            # Phase 2: Try 1-1 swap
            if len(edges) < 2:
                continue
            for _ in range(20):
                if not elapsed_ok():
                    break
                idx = random.randint(0, len(edges) - 1)
                u_rem, v_rem = edges[idx]
                
                cand_mask = adj_mask[:]
                cand_mask[u_rem] &= ~(1 << v_rem)
                cand_mask[v_rem] &= ~(1 << u_rem)
                cand_edges = edges[:idx] + edges[idx+1:]
                
                # Try to add a new edge
                missing = [(u, v) for u in range(n) for v in range(u+1, n) 
                          if not (cand_mask[u] & (1 << v))]
                random.shuffle(missing)
                
                swapped = False
                for u_add, v_add in missing[:60]:
                    if not elapsed_ok():
                        break
                    if not _can_add_edge(u_add, v_add, cand_mask):
                        continue
                    tmp_mask = cand_mask[:]
                    tmp_mask[u_add] |= (1 << v_add)
                    tmp_mask[v_add] |= (1 << u_add)
                    if _is_3_colorable(tmp_mask, n):
                        continue
                    
                    adj_mask = tmp_mask
                    edges = cand_edges + [[u_add, v_add]]
                    swapped = True
                    break
                
                if swapped:
                    improved = True
                    break
        
        return adj_mask, edges
    
    # Run multiple instances with different strategies
    for restart in range(200):
        if not elapsed_ok():
            break
        
        # Start with Groetzsch graph
        base_edges = _groetzsch_edges()
        adj_mask = _build_adj_mask(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free(adj_mask, n) or _is_3_colorable(adj_mask, n):
            continue
        
        # Generate pair order
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        if restart % 3 == 0:
            # Random order
            random.seed(restart * 7 + 13)
            random.shuffle(all_pairs)
        elif restart % 3 == 1:
            # Degree-based order (try high-degree vertices first)
            deg = [adj_mask[i].bit_count() for i in range(n)]
            all_pairs.sort(key=lambda p: -(deg[p[0]] + deg[p[1]]))
        else:
            # Random but biased toward lower-degree vertices
            random.seed(restart * 11 + 7)
            random.shuffle(all_pairs)
        
        # Greedy fill
        adj_mask, edges = greedy_fill(adj_mask, edges, all_pairs)
        
        # Local search
        adj_mask, edges = local_search(adj_mask, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
    
    # Final validation
    if best_edges:
        final_mask = _build_adj_mask(best_edges, n)
        assert _is_triangle_free(final_mask, n), "Not triangle-free"
        assert not _is_3_colorable(final_mask, n), "3-colorable"
        return best_edges
    else:
        return _groetzsch_edges()
