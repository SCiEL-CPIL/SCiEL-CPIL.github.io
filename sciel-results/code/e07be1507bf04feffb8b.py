import time
import random

# ---------- Exact 3-colorability via bitmasks ----------
def _is_3_colorable(adj_mask, n=16):
    """adj_mask: list of 16 int bitmasks. Returns True if graph is 3-colorable."""
    # Order vertices by descending degree
    order = sorted(range(n), key=lambda v: -bin(adj_mask[v]).count('1'))
    
    color = [-1] * n
    color_mask = [0, 0, 0]
    
    def backtrack(i):
        v = order[i]
        for c in range(3):
            if adj_mask[v] & color_mask[c]:
                continue
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
    """Groetzsch graph (Mycielski of C5) on vertices 0-10."""
    edges = []
    # C5: 0-1-2-3-4-0
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        edges.append([u, v])
    # Mycielski: add vertex 5 connected to 1,2,3,4
    # Wait, standard Mycielski: for each edge (u,v) in G, add edges (u',v) and (v',u)
    # where u' is the new vertex for u.
    # Actually, the Groetzsch graph is usually defined as:
    # Vertices 0-4 form C5.
    # Vertex 5 is connected to all of 0-4? No, that would make triangles.
    # Let's use the standard Mycielski construction.
    # G = C5 (vertices 0-4)
    # G' has vertices 0-4 (original) and 5-9 (copies) and 10 (special).
    # Edges:
    # 1. Original edges of C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # 2. For each edge (u,v) in G, add (u', v) and (v', u).
    #    So (5,1), (6,0), (6,2), (7,1), (7,3), (8,2), (8,4), (9,3), (9,0)
    # 3. Special vertex 10 connected to all copies 5-9: (10,5), (10,6), (10,7), (10,8), (10,9)
    edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    # Mycielski edges
    edges += [[5,1], [6,0], [6,2], [7,1], [7,3], [8,2], [8,4], [9,3], [9,0]]
    # Special vertex
    edges += [[10,5], [10,6], [10,7], [10,8], [10,9]]
    return edges

def construct():
    start = time.time()
    deadline = start + 115.0
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline
    
    def run_greedy(seed):
        random.seed(seed)
        base_edges = _groetzsch_edges()
        adj_mask = _build_adj_mask(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free(adj_mask, n) or _is_3_colorable(adj_mask, n):
            return adj_mask, edges, len(edges)
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if not _can_add_edge(u, v, adj_mask):
                    continue
                new_adj_mask = adj_mask[:]
                new_adj_mask[u] |= (1 << v)
                new_adj_mask[v] |= (1 << u)
                if _is_3_colorable(new_adj_mask, n):
                    continue
                adj_mask = new_adj_mask
                edges.append([u, v])
                changed = True
        
        # Local search: 1-1 swaps
        improved = True
        while improved and elapsed_ok():
            improved = False
            # Try to add any edge
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if not _can_add_edge(u, v, adj_mask):
                    continue
                new_adj_mask = adj_mask[:]
                new_adj_mask[u] |= (1 << v)
                new_adj_mask[v] |= (1 << u)
                if _is_3_colorable(new_adj_mask, n):
                    continue
                adj_mask = new_adj_mask
                edges.append([u, v])
                improved = True
                break
            
            if improved or not elapsed_ok():
                continue
            
            # Try 1-1 swap
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u_rem, v_rem = edges[idx]
            
            cand_adj_mask = adj_mask[:]
            cand_adj_mask[u_rem] &= ~(1 << v_rem)
            cand_adj_mask[v_rem] &= ~(1 << u_rem)
            cand_edges = edges[:idx] + edges[idx+1:]
            
            pairs_to_try = [(u, v) for u in range(n) for v in range(u+1, n) if not (cand_adj_mask[u] & (1 << v))]
            random.shuffle(pairs_to_try)
            
            swapped = False
            for u_add, v_add in pairs_to_try[:80]:
                if not elapsed_ok():
                    break
                if not _can_add_edge(u_add, v_add, cand_adj_mask):
                    continue
                tmp_adj_mask = cand_adj_mask[:]
                tmp_adj_mask[u_add] |= (1 << v_add)
                tmp_adj_mask[v_add] |= (1 << u_add)
                if _is_3_colorable(tmp_adj_mask, n):
                    continue
                
                adj_mask = tmp_adj_mask
                edges = cand_edges + [[u_add, v_add]]
                swapped = True
                break
            
            if swapped:
                improved = True
        
        return adj_mask, edges, len(edges)
    
    max_restarts = 100
    for i in range(max_restarts):
        if not elapsed_ok():
            break
        seed = i * 1000 + 42
        adj_mask, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
    
    if best_edges:
        final_adj_mask = _build_adj_mask(best_edges, n)
        assert _is_triangle_free(final_adj_mask, n)
        assert not _is_3_colorable(final_adj_mask, n)
        return best_edges
    else:
        return _groetzsch_edges()

