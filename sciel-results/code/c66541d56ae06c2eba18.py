import time

def construct():
    start_time = time.time()
    # Safety margin: The limit is 90s. Let's aim to finish by 85s to be safe.
    timeout = 85.0
    
    n = 256
    vertices = range(1, n + 1)
    
    # 1. Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    # We want the Maximum Independent Set in G, which is the Maximum Clique in G_complement.
    
    # adjacency bitmask for G (conflicts)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    # adj_comp[v] contains bits for all u != v such that (a*b) % (a+b) != 0
    adj_comp = [0] * (n + 1)
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        # Start with all vertices, remove v itself, remove conflict neighbors
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
        
    # Tomita Algorithm for Maximum Clique
    best_size = 0
    best_set_mask = 0
    
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

    # Optimized greedy coloring for upper bound
    # Uses bitset-based color tracking to avoid set operations
    def greedy_color(candidates_mask):
        # Get vertices and sort by degree descending
        verts = get_vertices(candidates_mask)
        
        # Calculate degrees within candidates
        degrees = []
        for v in verts:
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        # Sort by degree descending
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        # Color assignment using bitset tracking
        # color_bits[c] is a bitmask of vertices assigned color c
        color_bits = []
        max_color = 0
        vertex_color = [0] * (n + 1)
        
        for v in ordered_verts:
            # Find which colors are used by neighbors
            neighbor_mask = adj_comp[v] & candidates_mask
            # For each color, check if any neighbor has that color
            used_color_mask = 0
            for c_idx, c_mask in enumerate(color_bits):
                if neighbor_mask & c_mask:
                    used_color_mask |= (1 << c_idx)
            
            # Find smallest unused color
            c = 0
            while (used_color_mask >> c) & 1:
                c += 1
            
            # Assign color c
            if c >= len(color_bits):
                color_bits.append(0)
            color_bits[c] |= (1 << v)
            vertex_color[v] = c
            if c + 1 > max_color:
                max_color = c + 1
        
        # Create list of (vertex, color) sorted by color descending
        color_list = [(v, vertex_color[v] + 1) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        # Check timeout
        if time.time() - start_time > timeout:
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

    # Initial heuristic to get a good lower bound quickly
    init_candidates = all_vertices_mask
    init_clique_mask = 0
    init_size = 0
    while init_candidates:
        # Pick vertex with max degree in init_candidates
        verts = get_vertices(init_candidates)
        best_v = -1
        best_deg = -1
        for v in verts:
            deg = (adj_comp[v] & init_candidates).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if best_v == -1: break
        init_clique_mask |= (1 << best_v)
        init_size += 1
        init_candidates = init_candidates & adj_comp[best_v]
        
    best_size = init_size
    best_set_mask = init_clique_mask

    # Start the main search
    expand(all_vertices_mask, 0, 0)
    
    # Local search polish: try to improve the found clique
    if time.time() - start_time < timeout - 2:
        # Get current best clique as a set
        current_clique = set()
        m = best_set_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            current_clique.add(v)
            m ^= lsb
        
        # Try to add vertices one by one
        improved = True
        while improved and time.time() - start_time < timeout - 2:
            improved = False
            for v in range(1, n + 1):
                if v in current_clique:
                    continue
                # Check if v can be added (compatible with all in clique)
                can_add = True
                for u in current_clique:
                    if v != u and (adj_conflict[v] & (1 << u)):
                        can_add = False
                        break
                if can_add:
                    current_clique.add(v)
                    improved = True
        
        # Update best if improved
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_set_mask = 0
            for v in current_clique:
                best_set_mask |= (1 << v)
    
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
