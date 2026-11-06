def construct():
    start = time.time()
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    m = len(all_edges)  # 190 edges
    
    # Precompute all 4-cycles
    four_cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        four_cycles.append(((a,b), (b,c), (c,d), (a,d)))
        four_cycles.append(((a,b), (b,d), (c,d), (a,c)))
        four_cycles.append(((a,c), (b,c), (b,d), (a,d)))
    
    num_cycles = len(four_cycles)
    
    # Edge to index mapping
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Precompute for each edge, which 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(m)]
    for ci, cycle in enumerate(four_cycles):
        for e in cycle:
            ei = edge_idx[e]
            edge_to_cycles[ei].append(ci)
    
    # Precompute for each 4-cycle, the 4 edge indices
    cycle_edges = []
    for cycle in four_cycles:
        cycle_edges.append([edge_idx[e] for e in cycle])

    def try_construct(order):
        # color_of[i] is the color of edge i, or -1 if edge is not present
        color_of = np.full(m, -1, dtype=np.int32)
        active_edges = 0
        
        for ei in order:
            # Try to assign a color to edge ei
            valid_color = -1
            for c in range(num_colors):
                # Check if assigning color c to ei creates any non-rainbow 4-cycle
                is_valid = True
                for ci in edge_to_cycles[ei]:
                    ce = cycle_edges[ci]
                    cols = []
                    cycle_formed = True
                    for j in range(4):
                        if ce[j] == ei:
                            cols.append(c)
                        else:
                            other_c = color_of[ce[j]]
                            if other_c == -1:
                                cycle_formed = False
                                break
                            cols.append(other_c)
                    
                    if not cycle_formed:
                        continue
                    
                    if len(set(cols)) < 4:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_color = c
                    break
            
            if valid_color != -1:
                color_of[ei] = valid_color
                active_edges += 1
        
        return color_of, active_edges

    best_color_of = None
    best_active = 0
    
    # Run multiple random restarts
    while time.time() - start < DEADLINE:
        order = list(range(m))
        random.shuffle(order)
        
        color_of, active = try_construct(order)
        
        if active > best_active:
            best_active = active
            best_color_of = color_of
            if best_active == m:
                break

    # Build result
    result = []
    if best_color_of is not None:
        for i, (u, v) in enumerate(all_edges):
            c = best_color_of[i]
            if c != -1:
                result.append([u, v, int(c)])
    
    return result
