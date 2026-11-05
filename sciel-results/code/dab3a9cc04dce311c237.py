def construct():
    start_time = time.time()
    cycles = get_four_cycles()
    num_cycles = len(cycles)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Pre-compute for each edge, which cycles it participates in and its position
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e0, e1, e2, e3) in enumerate(cycles):
        for e in (e0, e1, e2, e3):
            edge_to_cycles[e].append(ci)
    
    edge_cycle_pos = [[] for _ in range(NUM_EDGES)]
    for ci, (e0, e1, e2, e3) in enumerate(cycles):
        for pos, e in enumerate((e0, e1, e2, e3)):
            edge_cycle_pos[e].append((ci, pos))
    
    cycle_edges = [tuple(c) for c in cycles]
    
    def try_assign_fast(colors, edge_idx, color_val):
        """Try assigning color_val to edge_idx, return True if valid. Optimized."""
        for ci, pos in edge_cycle_pos[edge_idx]:
            e0, e1, e2, e3 = cycle_edges[ci]
            c0 = colors[e0] if pos != 0 else color_val
            c1 = colors[e1] if pos != 1 else color_val
            c2 = colors[e2] if pos != 2 else color_val
            c3 = colors[e3] if pos != 3 else color_val
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
        
    random.seed(42)
    
    # Multi-restart strategy
    # Each restart: 
    # 1. Start with all 0s
    # 2. Repeatedly pass through shuffled edges, trying to assign the NEXT new color
    # 3. Stop when a full pass yields no new colors
    
    for restart in range(100):
        if time.time() - start_time > DEADLINE:
            break
            
        colors = [0] * NUM_EDGES
        used_colors = {0}
        curr_count = 1
        
        # Multi-pass greedy
        progress = True
        while progress:
            progress = False
            edges = list(range(NUM_EDGES))
            random.shuffle(edges)
            
            for e in edges:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Try to assign the next new color
                new_color = max(used_colors) + 1
                if try_assign_fast(colors, e, new_color):
                    colors[e] = new_color
                    used_colors.add(new_color)
                    curr_count += 1
                    progress = True
            
            if time.time() - start_time > DEADLINE:
                break
        
        if curr_count > best_count:
            best_count = curr_count
            best_colors = colors[:]
            
    return best_colors
