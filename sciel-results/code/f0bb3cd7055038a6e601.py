import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute triangle conflicts: for each edge (u,v), which edges (a,b) would form a triangle
    # Edge (u,v) and (a,b) form a triangle if they share a vertex and the third edge exists
    # Actually, adding (u,v) creates a triangle if there's a vertex w such that (u,w) and (v,w) are both present
    # So for each pair of edges, we can't precompute statically. Instead, we'll use bitmask checks.
    
    # Fast 3-colorability check using bitmasks with strong pruning
    def is_3_colorable(adj_mask):
        """Fast 3-colorability check with MRV + forward checking + unit propagation."""
        color = [-1] * n
        # adj_mask[v] is bitmask of neighbors of v
        
        def get_available_colors(v):
            """Get bitmask of available colors for vertex v."""
            used = 0
            m = adj_mask[v]
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                if color[u] >= 0:
                    used |= (1 << color[u])
                m ^= lsb
            return ~(used & 0b111) & 0b111  # 3 bits, only colors 0,1,2
        
        def backtrack():
            # Find uncolored vertex with fewest available colors (MRV)
            best_v = -1
            best_avail_count = 4
            for v in range(n):
                if color[v] == -1:
                    avail = get_available_colors(v)
                    count = bin(avail).count('1')
                    if count < best_avail_count:
                        best_avail_count = count
                        best_v = v
                        if count == 0:
                            return False
                        if count == 1:
                            break
            
            if best_v == -1:
                return True
            
            v = best_v
            avail = get_available_colors(v)
            
            # Try colors in order of most constrained neighbors first
            # For simplicity, try 0,1,2
            for c in range(3):
                if avail & (1 << c):
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = -1
            return False
        
        return backtrack()
    
    def edge_count(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def is_triangle_free(adj_mask):
        """Check if graph is triangle-free using bitmasks."""
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                    if common:
                        return False
        return True
    
    # Build initial graph: Mycielski of C5 (Grötzsch graph) on 11 vertices
    adj_mask = [0] * n
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(adj_mask, i, (i+1) % 5)
    
    # Mycielski: vertices 5-9 are copies, vertex 10 is apex
    for i in range(5):
        u, v = i, (i+1) % 5
        add_edge(adj_mask, u, 5+v)
        add_edge(adj_mask, v, 5+u)
    
    for i in range(5):
        add_edge(adj_mask, 5+i, 10)
    
    # Add vertices 11-15 by connecting to large independent sets
    def find_large_independent_set(adj_mask, exclude=None):
        """Find a large independent set greedily."""
        if exclude is None:
            exclude = set()
        
        remaining = [v for v in range(n) if v not in exclude]
        # Sort by degree ascending
        remaining.sort(key=lambda v: bin(adj_mask[v]).count('1'))
        
        mis = []
        used = 0
        for v in remaining:
            if not (adj_mask[v] & used):
                mis.append(v)
                used |= (1 << v)
        
        return mis
    
    for new_v in range(11, 16):
        mis = find_large_independent_set(adj_mask)
        for v in mis:
            if v != new_v:
                # Check triangle-free
                common = adj_mask[new_v] & adj_mask[v] & ~(1 << new_v) & ~(1 << v)
                if not common:
                    add_edge(adj_mask, new_v, v)
    
    best_adj = list(adj_mask)
    best_count = edge_count(best_adj)
    
    # Greedy edge addition with time budget
    def greedy_add(adj_mask, max_time):
        changed = True
        while changed and time.time() < max_time:
            changed = False
            edges = list(all_edges)
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > max_time:
                    break
                if adj_mask[u] & (1 << v):
                    continue
                # Check triangle-free
                common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                # Try adding
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                add_edge(adj_mask, u, v)
                changed = True
        return adj_mask
    
    adj_mask = greedy_add(adj_mask, deadline - 50)
    current_count = edge_count(adj_mask)
    if current_count > best_count:
        best_count = current_count
        best_adj = list(adj_mask)
    
    # Local search: remove one edge, try to add one or more edges
    def local_search(adj_mask, max_time):
        current = list(adj_mask)
        current_count = edge_count(current)
        best_local = list(current)
        best_local_count = current_count
        
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < max_time and no_improve < max_no_improve:
            existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if current[i] & (1 << j)]
            if not existing:
                break
            
            ui, vi = random.choice(existing)
            remove_edge(current, ui, vi)
            
            # Try to add multiple edges
            added_any = False
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                         if not (current[i] & (1 << j)) and (i,j) != (ui, vi)]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > max_time:
                    break
                common = current[u] & current[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                new_adj = list(current)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                add_edge(current, u, v)
                added_any = True
                # Continue trying to add more edges
                # Refresh candidates
                candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                             if not (current[i] & (1 << j))]
                random.shuffle(candidates)
            
            if not added_any:
                add_edge(current, ui, vi)
                no_improve += 1
                continue
            
            new_count = edge_count(current)
            if new_count > best_local_count:
                best_local_count = new_count
                best_local = list(current)
                no_improve = 0
            else:
                no_improve += 1
        
        return best_local, best_local_count
    
    adj_mask, local_count = local_search(adj_mask, deadline - 25)
    if local_count > best_count:
        best_count = local_count
        best_adj = list(adj_mask)
    
    # Multi-start with perturbation
    for restart in range(8):
        if time.time() > deadline - 20:
            break
        
        current = list(best_adj)
        existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if current[i] & (1 << j)]
        random.shuffle(existing)
        num_remove = min(4, len(existing))
        for edge in existing[:num_remove]:
            remove_edge(current, edge[0], edge[1])
        
        current = greedy_add(current, deadline - 20 - restart)
        current_count = edge_count(current)
        
        current, ls_count = local_search(current, deadline - 15 - restart)
        if ls_count > best_count:
            best_count = ls_count
            best_adj = list(current)
    
    # Final systematic polish: try every possible edge addition
    final_adj = list(best_adj)
    for u, v in all_edges:
        if time.time() > deadline - 1:
            break
        if final_adj[u] & (1 << v):
            continue
        common = final_adj[u] & final_adj[v] & ~(1 << u) & ~(1 << v)
        if common:
            continue
        new_adj = list(final_adj)
        add_edge(new_adj, u, v)
        if is_3_colorable(new_adj):
            continue
        add_edge(final_adj, u, v)
    
    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj
    
    # Return result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
