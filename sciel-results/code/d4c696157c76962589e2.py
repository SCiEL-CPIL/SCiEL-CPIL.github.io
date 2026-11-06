import time

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin from 60s limit
    
    n = 256
    vertices = range(1, n + 1)
    
    # Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    adj_comp = [0] * (n + 1)
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
        
    best_size = 0
    best_set_mask = 0
    
    def popcount(x):
        return x.bit_count()
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts
    
    def iter_vertices(mask):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            yield v
            m ^= lsb

    def greedy_color(candidates_mask):
        """Fast greedy coloring using bitmask color classes."""
        uncolored = candidates_mask
        verts = list(iter_vertices(candidates_mask))
        
        # Sort by degree in candidate subgraph descending
        degrees = []
        for v in verts:
            deg = popcount(adj_comp[v] & candidates_mask)
            degrees.append((deg, v))
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        # Color classes as bitmasks
        color_classes = []
        max_color = 0
        
        for v in ordered_verts:
            v_bit = 1 << v
            v_neighbors = adj_comp[v] & uncolored  # neighbors already colored
            
            # Find first color class where v has no neighbor
            placed = False
            for c_idx, c_mask in enumerate(color_classes):
                if not (c_mask & v_neighbors):
                    color_classes[c_idx] |= v_bit
                    placed = True
                    break
            
            if not placed:
                color_classes.append(v_bit)
                max_color += 1
            
            uncolored &= ~v_bit
        
        # Build color list sorted by color descending
        color_list = []
        for c_idx, c_mask in enumerate(color_classes):
            c = len(color_classes) - c_idx  # higher index = higher color
            for v in iter_vertices(c_mask):
                color_list.append((v, c))
        
        color_list.sort(key=lambda x: x[1], reverse=True)
        return color_list, max_color

    # Greedy initial solution for better lower bound
    def greedy_initial():
        """Get a good initial clique using greedy heuristic."""
        nonlocal best_size, best_set_mask
        
        remaining = all_vertices_mask
        current_mask = 0
        current_size = 0
        
        while remaining:
            # Pick vertex with max degree in remaining
            best_v = None
            best_deg = -1
            for v in iter_vertices(remaining):
                deg = popcount(adj_comp[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            current_mask |= (1 << best_v)
            current_size += 1
            remaining &= adj_comp[best_v]
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_mask
        
        return best_size

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
        
        # Quick upper bound: number of candidates
        n_cands = popcount(candidates_mask)
        if current_size + n_cands <= best_size:
            return
        
        # Upper bound via coloring
        color_list, max_color = greedy_color(candidates_mask)
        
        if current_size + max_color <= best_size:
            return
            
        P = candidates_mask
        
        for v, c in color_list:
            if current_size + c <= best_size:
                break
                
            if not (P & (1 << v)):
                continue
                
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            P &= ~(1 << v)

    # Get initial lower bound
    greedy_initial()
    
    # Start the exact search
    expand(all_vertices_mask, 0, 0)
    
    # Extract result
    result_set = set()
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        result_set.add(v)
        mask ^= lsb
        
    if not result_set:
        result_set = {1}
        
    return list(result_set)

# Run once to verify it works and time it
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
