import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # 1. Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    adj_conflict = [0] * (N + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    adj_comp = [0] * (N + 1)
    all_vertices_mask = 0
    for i in range(1, N + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
    
    # Helper to get list of vertices from a bitmask
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts
    
    # Greedy coloring for upper bound - optimized version
    def greedy_color(candidates_mask):
        verts = get_vertices(candidates_mask)
        if not verts:
            return [], 0
        
        # Calculate degrees within candidates
        degrees = []
        for v in verts:
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        # Sort by degree descending
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        # Use bitmask-based color tracking
        color_masks = [0]
        max_color = 0
        vertex_colors = {}
        
        for v in ordered_verts:
            # Find smallest color c such that no neighbor of v has color c
            c = 1
            while True:
                if c < len(color_masks):
                    if (adj_comp[v] & color_masks[c]) == 0:
                        break
                else:
                    color_masks.append(0)
                    break
                c += 1
            
            vertex_colors[v] = c
            color_masks[c] |= (1 << v)
            if c > max_color:
                max_color = c
        
        # Create list of (vertex, color) sorted by color descending
        color_list = [(v, vertex_colors[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color
    
    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask
        
        # Check timeout
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Get upper bound via coloring
        color_list, max_color = greedy_color(candidates_mask)
        
        # If the upper bound doesn't exceed best, prune
        if current_size + max_color <= best_size:
            return
            
        P = candidates_mask
        
        for v, c in color_list:
            if not (P & (1 << v)):
                continue
                
            # Prune if bound is too low
            if current_size + c <= best_size:
                break
                
            # Recurse: Pick v
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            # Don't pick v: Remove v from P
            P &= ~(1 << v)
    
    # Initial heuristic: try multiple greedy strategies for better lower bound
    def greedy_clique(start_mask):
        candidates = start_mask
        clique_mask = 0
        size = 0
        while candidates:
            verts = get_vertices(candidates)
            best_v = -1
            best_deg = -1
            for v in verts:
                deg = (adj_comp[v] & candidates).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                break
            clique_mask |= (1 << best_v)
            size += 1
            candidates = candidates & adj_comp[best_v]
        return size, clique_mask
    
    # Strategy 1: greedy from full set
    s1, m1 = greedy_clique(all_vertices_mask)
    
    # Strategy 2: try removing each vertex and greedy from remaining
    best_s = s1
    best_m = m1
    for v in vertices:
        if time.time() > deadline - 5:
            break
        mask_without_v = all_vertices_mask & ~(1 << v)
        s, m = greedy_clique(mask_without_v)
        if s > best_s:
            best_s = s
            best_m = m
    
    # Strategy 3: Use structural insight - divisor-based cliques
    # For each k, numbers of the form k*d where d|k^2, d>1 form a clique in conflict graph
    # Their complement in [1,256] forms an independent set
    for k in range(1, N + 1):
        if time.time() > deadline - 3:
            break
        # Find all d such that d | k^2 and d > 1
        k_sq = k * k
        divisors = []
        for d in range(2, int(k_sq**0.5) + 1):
            if k_sq % d == 0:
                divisors.append(d)
                other = k_sq // d
                if other != d and other > 1:
                    divisors.append(other)
        
        # Numbers of form k*d for d in divisors, within [1, N]
        clique_nodes = set()
        for d in divisors:
            val = k * d
            if 1 <= val <= N:
                clique_nodes.add(val)
        
        if len(clique_nodes) >= 2:
            # Independent set is all vertices NOT in clique_nodes
            independent_mask = all_vertices_mask
            for node in clique_nodes:
                independent_mask &= ~(1 << node)
            
            # Verify it's actually an independent set (no conflicts)
            valid = True
            verts_in_set = get_vertices(independent_mask)
            for i, a in enumerate(verts_in_set):
                for b in verts_in_set[i+1:]:
                    if (a * b) % (a + b) == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                s = len(verts_in_set)
                if s > best_s:
                    best_s = s
                    best_m = independent_mask
    
    best_size = best_s
    best_set_mask = best_m

    # Start the search
    expand(all_vertices_mask, 0, 0)
    
    # Extract result
    result_set = set()
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        result_set.add(v)
        mask ^= lsb
        
    return list(result_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
