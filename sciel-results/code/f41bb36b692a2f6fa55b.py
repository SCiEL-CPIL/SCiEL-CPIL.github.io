import time
import random
import numpy as np

def _compute_max_book_bitmask(adj_list, n=40):
    """Compute max book size using bitmasks. adj_list[i] is a bitmask of neighbors of i."""
    max_book = 0
    for i in range(n):
        mask_i = adj_list[i]
        m = mask_i
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            m ^= lsb
            if j > i:
                common = bin(adj_list[i] & adj_list[j]).count('1')
                if common > max_book:
                    max_book = common
    return max_book

def _compute_max_book_affected(adj_list, n=40, affected_vertices=None):
    """Compute max book size, but only for edges where at least one endpoint is in affected_vertices.
    Returns (max_affected, all_max_estimate). For speed, we track global max separately."""
    if affected_vertices is None:
        return _compute_max_book_bitmask(adj_list, n)
    max_book = 0
    affected_set = set(affected_vertices)
    for i in range(n):
        if i not in affected_set:
            continue
        mask_i = adj_list[i]
        m = mask_i
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            m ^= lsb
            if j > i:
                common = bin(adj_list[i] & adj_list[j]).count('1')
                if common > max_book:
                    max_book = common
    return max_book

def _build_adj_list(adj_mask, n=40):
    """Build bitmask adjacency list from boolean adjacency matrix."""
    adj_list = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i, j]:
                adj_list[i] |= (1 << j)
                adj_list[j] |= (1 << i)
    return adj_list

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    best_max = 999
    best_adj = None
    
    random.seed(12345)
    
    configurations = []
    
    # Config 1: K20,20 + 1 edge (baseline, book=20)
    adj1 = np.zeros((n, n), dtype=bool)
    for a in range(20):
        for b in range(20, 40):
            adj1[a, b] = True
            adj1[b, a] = True
    adj1[0, 1] = True
    adj1[1, 0] = True
    configurations.append(adj1.copy())
    
    # Config 2: Split into two groups of 20, but with a more balanced intra-group structure
    for seed_val in [1, 2, 3, 4, 5]:
        random.seed(seed_val)
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        possible_intra = [(a, b) for a in range(20) for b in range(a+1, 20)]
        possible_intra += [(a, b) for a in range(20, 40) for b in range(a+1, 40)]
        u, v = random.choice(possible_intra)
        adj[u, v] = True
        adj[v, u] = True
        configurations.append(adj.copy())
    
    # Config 3: Try a 3-partite structure: 13, 13, 14
    sizes = [13, 13, 14]
    adj3 = np.zeros((n, n), dtype=bool)
    part_starts = [0, 13, 26]
    for pi in range(3):
        for pj in range(pi+1, 3):
            for i in range(part_starts[pi], part_starts[pi]+sizes[pi]):
                for j in range(part_starts[pj], part_starts[pj]+sizes[pj]):
                    adj3[i, j] = True
                    adj3[j, i] = True
    edge_count = np.sum(adj3) // 2
    random.seed(77)
    intra_candidates = []
    for pi in range(3):
        start = part_starts[pi]
        end = start + sizes[pi]
        for i in range(start, end):
            for j in range(i+1, end):
                intra_candidates.append((i, j))
    random.shuffle(intra_candidates)
    for u, v in intra_candidates:
        if edge_count >= target_edges:
            break
        adj3[u, v] = True
        adj3[v, u] = True
        edge_count += 1
    if edge_count == target_edges:
        configurations.append(adj3.copy())
    
    # Config 4: Regular-ish graph with degree ~20 (circulant)
    adj4 = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for k in range(1, 11):
            adj4[i, (i+k) % n] = True
            adj4[(i+k) % n, i] = True
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj4[i, j]]
    if non_edges:
        u, v = non_edges[0]
        adj4[u, v] = True
        adj4[v, u] = True
    configurations.append(adj4.copy())
    
    # Config 5: 4-partite: 10,10,10,10
    adj5 = np.zeros((n, n), dtype=bool)
    part_starts5 = [0, 10, 20, 30]
    for pi in range(4):
        for pj in range(pi+1, 4):
            for i in range(part_starts5[pi], part_starts5[pi]+10):
                for j in range(part_starts5[pj], part_starts5[pj]+10):
                    adj5[i, j] = True
                    adj5[j, i] = True
    edge_count5 = np.sum(adj5) // 2
    random.seed(88)
    intra_candidates5 = []
    for pi in range(4):
        start = part_starts5[pi]
        for i in range(start, start+10):
            for j in range(i+1, start+10):
                intra_candidates5.append((i, j))
    random.shuffle(intra_candidates5)
    for u, v in intra_candidates5:
        if edge_count5 >= target_edges:
            break
        adj5[u, v] = True
        adj5[v, u] = True
        edge_count5 += 1
    if edge_count5 == target_edges:
        configurations.append(adj5.copy())
    
    # Config 6: 5-partite: 8,8,8,8,8
    adj6 = np.zeros((n, n), dtype=bool)
    part_starts6 = [0, 8, 16, 24, 32]
    for pi in range(5):
        for pj in range(pi+1, 5):
            for i in range(part_starts6[pi], part_starts6[pi]+8):
                for j in range(part_starts6[pj], part_starts6[pj]+8):
                    adj6[i, j] = True
                    adj6[j, i] = True
    edge_count6 = np.sum(adj6) // 2
    random.seed(99)
    intra_candidates6 = []
    for pi in range(5):
        start = part_starts6[pi]
        for i in range(start, start+8):
            for j in range(i+1, start+8):
                intra_candidates6.append((i, j))
    random.shuffle(intra_candidates6)
    for u, v in intra_candidates6:
        if edge_count6 >= target_edges:
            break
        adj6[u, v] = True
        adj6[v, u] = True
        edge_count6 += 1
    if edge_count6 == target_edges:
        configurations.append(adj6.copy())
    
    # Hill climb each configuration using bitmask-based book computation
    for init_adj in configurations:
        adj = init_adj.copy()
        adj_list = _build_adj_list(adj, n)
        current_max = _compute_max_book_bitmask(adj_list, n)
        
        if current_max < best_max:
            best_max = current_max
            best_adj = adj.copy()
        
        no_improve = 0
        max_no_improve = 30000
        
        # Pre-compute all non-edges for faster candidate selection
        all_non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    all_non_edges.append((i, j))
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Get all edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            
            if len(edges) != target_edges:
                break
            
            # Pick random edge to remove
            rem_u, rem_v = random.choice(edges)
            
            # Try several candidate new edges
            improved = False
            for attempt in range(8):
                if not all_non_edges:
                    break
                new_u, new_v = random.choice(all_non_edges)
                if new_u == rem_u and new_v == rem_v:
                    continue
                if adj[new_u, new_v]:
                    continue
                
                # Tentative swap
                adj[rem_u, rem_v] = False
                adj[rem_v, rem_u] = False
                adj[new_u, new_v] = True
                adj[new_v, new_u] = True
                adj_list[rem_u] &= ~(1 << rem_v)
                adj_list[rem_v] &= ~(1 << rem_u)
                adj_list[new_u] |= (1 << new_v)
                adj_list[new_v] |= (1 << new_u)
                
                # Only check edges affected by the swap
                affected = [rem_u, rem_v, new_u, new_v]
                new_max_affected = _compute_max_book_affected(adj_list, n, affected)
                
                # If the affected max is still >= current_max, full check needed
                if new_max_affected < current_max:
                    # Safe to accept - affected edges can't have book >= current_max
                    new_max = new_max_affected
                    # But we need to verify no other edge has higher book
                    # Actually, if affected < current_max, other edges unchanged so their book <= old max
                    # So new_max = max(new_max_affected, old_max_of_unaffected)
                    # Old unaffected edges had book <= current_max, so new_max = max(new_max_affected, min(current_max, ...))
                    # Simpler: just do full check if close
                    new_max = _compute_max_book_bitmask(adj_list, n)
                else:
                    new_max = _compute_max_book_bitmask(adj_list, n)
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_adj = adj.copy()
                    no_improve = 0
                    improved = True
                    # Update non-edges list
                    all_non_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if not adj[i, j]:
                                all_non_edges.append((i, j))
                    break
                else:
                    # Revert
                    adj[new_u, new_v] = False
                    adj[new_v, new_u] = False
                    adj[rem_u, rem_v] = True
                    adj[rem_v, rem_u] = True
                    adj_list[rem_u] |= (1 << rem_v)
                    adj_list[rem_v] |= (1 << rem_u)
                    adj_list[new_u] &= ~(1 << new_v)
                    adj_list[new_v] &= ~(1 << new_u)
            
            if not improved:
                no_improve += 1
            
            if current_max <= 6:
                break
        
        if time.time() > deadline:
            break
    
    if best_adj is None:
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        adj[0, 1] = True
        adj[1, 0] = True
        best_adj = adj
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
    
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
