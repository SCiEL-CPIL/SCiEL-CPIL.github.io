import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin from 90s limit

    n = 256
    vertices = range(1, n + 1)

    # 1. Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    # We will solve for Maximum Independent Set in G.
    # This is equivalent to Maximum Clique in G_complement.
    
    # adjacency bitmask for G (conflicts)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    # A vertex u is a neighbor of v in G_comp if u != v and u is NOT a neighbor of v in G.
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
    # We use bitmasks.
    best_size = 0
    best_set_mask = 0

    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts

    # Greedy coloring for upper bound
    def greedy_color(candidates_mask):
        colors = {}
        verts = get_vertices(candidates_mask)
        
        degrees = []
        for v in verts:
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        max_color = 0
        colored_mask = 0
        
        for v in ordered_verts:
            used_colors = set()
            m = adj_comp[v] & colored_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                used_colors.add(colors[u])
                m ^= lsb
                    
            c = 1
            while c in used_colors:
                c += 1
                
            colors[v] = c
            if c > max_color:
                max_color = c
            colored_mask |= (1 << v)
            
        color_list = [(v, colors[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
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

    initial_candidates = all_vertices_mask
    expand(initial_candidates, 0, 0)
    
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
