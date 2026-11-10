import math

def construct():
    N = 512
    limit = 512
    
    # Build valid_adj: adj[i] has bit j set if lcm(i, j) <= 512
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= limit:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
                
    best_size = 0
    best_set_mask = 0
    
    def greedy_color_bound(P_mask):
        # Returns upper bound on clique size in Valid Graph induced by P_mask
        # by computing a greedy coloring of the Valid Graph induced by P_mask.
        # Note: We are coloring the VALID graph.
        # A clique in Valid Graph has all distinct colors.
        # So num_colors >= clique_size.
        
        if P_mask == 0:
            return 0
            
        # Extract vertices and sort by degree in P_mask descending
        # This helps in finding a compact coloring (fewer colors)
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            val = idx + 1
            # Degree in P_mask
            deg = (adj[val] & P_mask).bit_count()
            verts.append((deg, val))
            temp ^= lsb
            
        verts.sort(reverse=True)
        
        colors = [] # List of bitmasks representing color classes
        
        for deg, val in verts:
            v_bit = 1 << (val - 1)
            placed = False
            # Try to place in existing color class
            for i, c_mask in enumerate(colors):
                # Check if v is non-adjacent to all in c_mask in Valid Graph
                # i.e., (adj[val] & c_mask) == 0
                if (adj[val] & c_mask) == 0:
                    colors[i] |= v_bit
                    placed = True
                    break
            if not placed:
                colors.append(v_bit)
                
        return len(colors)

    def expand(P_mask, current_size, current_set_mask):
        nonlocal best_size, best_set_mask
        
        if current_size > best_size:
            best_size = current_size
            best_set_mask = current_set_mask
            
        if P_mask == 0:
            return
            
        # Bound
        bound = greedy_color_bound(P_mask)
        if current_size + bound <= best_size:
            return
            
        # Order vertices for iteration
        # Sort by degree in P_mask descending
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            val = idx + 1
            deg = (adj[val] & P_mask).bit_count()
            verts.append((deg, val, lsb))
            temp ^= lsb
        verts.sort(reverse=True)
        
        # Iterate
        for deg, val, v_bit in verts:
            # Remove v from P
            P_mask ^= v_bit
            
            # Optional: Check if remaining P can improve
            # Simple count check
            if current_size + 1 + P_mask.bit_count() <= best_size:
                continue # Or return? 
                # In Tomita, if we process in order of decreasing color, we can return.
                # Here we just skip if simple bound fails.
                
            new_P = P_mask & adj[val]
            expand(new_P, current_size + 1, current_set_mask | v_bit)

    all_vertices = (1 << N) - 1
    expand(all_vertices, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
