N = 16

def construct():
    # Base solution from current program
    colors = [0] * (N * (N - 1) // 2)
    
    triangles = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14]
    ]
    
    current_color = 1
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)
        
    for tri in triangles:
        v1, v2, v3 = tri
        colors[get_idx(v1, v2)] = current_color; current_color += 1
        colors[get_idx(v2, v3)] = current_color; current_color += 1
        colors[get_idx(v1, v3)] = current_color; current_color += 1
        
    # Now try to add colors to 0-edges
    # Precompute 4-cycles for fast checking
    # A 4-cycle is a set of 4 edges.
    # We can check if assigning color C to edge (u,v) creates a rainbow cycle.
    # This happens if there exists a 4-cycle containing (u,v) where the other 3 edges have distinct colors, all different from C.
    
    # To speed up, we can just check all 4-cycles containing (u,v).
    # Number of 4-cycles containing a specific edge (u,v) is (N-2)*(N-3)/2?
    # Vertices w, x distinct from u,v.
    # Cycle u-v-w-x-u or u-v-x-w-u.
    # So 2 * C(N-2, 2) = (N-2)(N-3) cycles?
    # For N=16, 14*13 = 182 cycles per edge.
    # 105 edges * 182 cycles * 4 checks = ~76k ops. Very fast.
    
    import itertools
    
    # List of all 4-cycles (as sets of 4 edge indices)
    # We only need to check cycles containing the edge we are coloring.
    
    # Optimization: Precompute for each edge index, the list of 4-cycles (as list of 4 edge indices) it belongs to.
    edge_cycles = [[] for _ in range(120)]
    
    for combo in itertools.combinations(range(N), 4):
        a, b, c, d = combo
        # 3 cycles
        # 1: a-b-c-d-a
        e1 = get_idx(a, b); e2 = get_idx(b, c); e3 = get_idx(c, d); e4 = get_idx(d, a)
        cycle_edges = [e1, e2, e3, e4]
        for e in cycle_edges:
            edge_cycles[e].append(cycle_edges)
            
        # 2: a-b-d-c-a
        e1 = get_idx(a, b); e2 = get_idx(b, d); e3 = get_idx(d, c); e4 = get_idx(c, a)
        cycle_edges = [e1, e2, e3, e4]
        for e in cycle_edges:
            edge_cycles[e].append(cycle_edges)
            
        # 3: a-c-b-d-a
        e1 = get_idx(a, c); e2 = get_idx(c, b); e3 = get_idx(b, d); e4 = get_idx(d, a)
        cycle_edges = [e1, e2, e3, e4]
        for e in cycle_edges:
            edge_cycles[e].append(cycle_edges)
            
    # Now iterate through edges that are 0
    # We want to assign new colors.
    # We should use a new color each time to maximize count.
    
    for i in range(120):
        if colors[i] == 0:
            new_color = current_color
            # Check if assigning new_color to edge i creates a rainbow 4-cycle
            valid = True
            for cycle in edge_cycles[i]:
                # cycle is a list of 4 edge indices
                # One of them is i
                # Get colors of the other 3
                other_colors = []
                for e in cycle:
                    if e == i:
                        continue
                    other_colors.append(colors[e])
                # The colors in the cycle will be new_color + other_colors
                # If all 4 are distinct, it's a rainbow cycle.
                # Since new_color is new (not used before? No, we are incrementing),
                # it is distinct from all previous colors.
                # So we just need to check if other_colors has 3 distinct values.
                if len(set(other_colors)) == 3:
                    valid = False
                    break
            
            if valid:
                colors[i] = new_color
                current_color += 1
                
    return colors
