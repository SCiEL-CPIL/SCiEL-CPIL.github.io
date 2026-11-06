import time

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def has_c6_in_color(adj_bitmasks, n, edge_u, edge_v, color):
        adj_u = adj_bitmasks[color][edge_u]
        adj_v = adj_bitmasks[color][edge_v]
        
        n_u_list = []
        mask_u = adj_u
        while mask_u:
            lsb = mask_u & -mask_u
            x1 = lsb.bit_length() - 1
            n_u_list.append(x1)
            mask_u ^= lsb
            
        n_v_list = []
        mask_v = adj_v
        while mask_v:
            lsb = mask_v & -mask_v
            x4 = lsb.bit_length() - 1
            n_v_list.append(x4)
            mask_v ^= lsb
            
        if len(n_u_list) < 2 or len(n_v_list) < 2:
            return False

        for x1 in n_u_list:
            for x4 in n_v_list:
                if x1 == x4:
                    continue
                
                adj_x1 = adj_bitmasks[color][x1]
                adj_x4 = adj_bitmasks[color][x4]
                
                mask_x1 = adj_x1 & ~(1 << edge_u) & ~(1 << x4)
                mask_x4 = adj_x4 & ~(1 << edge_v) & ~(1 << x1)
                
                temp_mask_x1 = mask_x1
                while temp_mask_x1:
                    lsb = temp_mask_x1 & -temp_mask_x1
                    x2 = lsb.bit_length() - 1
                    temp_mask_x1 ^= lsb
                    
                    if adj_bitmasks[color][x2] & mask_x4:
                        return True
                        
        return False

    def solve_with_time(n, time_limit):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        adj_bitmasks = [[0] * n for _ in range(4)]
        
        # used_colors[0] = True if color 0 is used, etc.
        # This helps with symmetry breaking.
        
        def bt(edge_idx, colors, used_colors):
            if time.time() > time_limit:
                return False, None
                
            if edge_idx == num_edges:
                return True, colors[:]
            
            u, v = edges[edge_idx]
            
            # Determine allowed colors based on symmetry
            # If only color 0 is used, we can map 1->1, 2->2, 3->3 but they are symmetric.
            # Actually, we can just fix the first edge (0,1) to 0.
            # Then for subsequent edges, we try colors.
            # To break symmetry:
            # When we assign a new color (not seen before), we can always map it to the smallest unused index?
            # No, because the colors are fixed 0..3.
            # Better: 
            # Edge 0: fix to 0.
            # Edge 1: try 0, 1. (If 2 or 3, swap with 1)
            # Edge 2: if colors {0,1} used, try 0,1,2. (If 3, swap with 2)
            # If colors {0} used, try 0,1.
            
            if edge_idx == 0:
                candidates = [0]
            else:
                candidates = []
                for c in range(4):
                    if not used_colors[c]:
                        # If this is a new color, we can restrict it to be the smallest available "new" color?
                        # No, we just need to avoid permutations of used colors.
                        # If used_colors has {0,1}, and we try 2, it's distinct from 0,1.
                        # If we try 3, it's also distinct. But 2 and 3 are symmetric.
                        # So we only try the smallest new color.
                        # But we also need to try used colors.
                        pass
                # Simpler symmetry breaking:
                # If we haven't used color c, we can only use the smallest such c?
                # No, that's too restrictive if we need multiple new colors.
                
                # Let's just try all 4, but prune if:
                # If c is new, and there exists a smaller new color c' < c, skip c?
                # Yes. If we are introducing a new color, we must introduce them in order 0,1,2,3.
                # But 0 is already used. So new colors must be 1, then 2, then 3.
                
                for c in range(4):
                    if has_c6_in_color(adj_bitmasks, n, u, v, c):
                        continue
                    
                    if not used_colors[c]:
                        # Check if we can use this new color
                        # Only allow if no smaller new color is available
                        # i.e., all colors < c must be used
                        can_use = True
                        for k in range(c):
                            if not used_colors[k]:
                                can_use = False
                                break
                        if not can_use:
                            continue
                    
                    candidates.append(c)
            
            for c in candidates:
                adj_bitmasks[c][u] |= (1 << v)
                adj_bitmasks[c][v] |= (1 << u)
                used_colors[c] = True
                colors.append(c)
                
                ok, res = bt(edge_idx + 1, colors, used_colors)
                
                if ok:
                    return True, res
                
                adj_bitmasks[c][u] &= ~(1 << v)
                adj_bitmasks[c][v] &= ~(1 << u)
                used_colors[c] = False
                colors.pop()
            
            return False, None

        colors = []
        used_colors = [False] * 4
        ok, res = bt(0, colors, used_colors)
        
        if ok:
            return res
        else:
            return None

    # Try n=18
    res_18 = solve_with_time(18, start_time + 30)
    if res_18 is not None:
        return {'n': 18, 'colors': res_18}
        
    # Fallback to n=17
    n = 17
    dist_assign = [0, 1, 2, 3, 1, 0, 3, 2]
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d_min = min(d, n - d)
            c = dist_assign[d_min - 1]
            colors.append(c)
            
    return {'n': 17, 'colors': colors}
