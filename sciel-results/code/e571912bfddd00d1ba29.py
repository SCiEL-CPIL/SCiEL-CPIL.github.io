import random
import time
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute 4-cycles
    # A 4-cycle is defined by 4 vertices.
    # There are 3 cycles in each K4.
    # We store the list of 4 edge indices for each cycle.
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        # Edges in the K4
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        # Cycle 2: v1-v2-v4-v3-v1
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        # Cycle 3: v1-v3-v2-v4-v1
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)

    # Initialize colors
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    # Helper to check validity of a constraint given current colors and a proposed change
    # Since we only change one edge, we can optimize.
    
    best_num_colors = 1
    
    while time.time() < deadline:
        # Try to introduce a new color
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        
        # Pick a random color, prefer unused ones if possible
        # To maximize, we should try to use colors not in used_colors
        # But if all 119 are used, we just maintain.
        if len(used_colors) < 120:
            unused = [c for c in range(120) if c not in used_colors]
            if unused:
                c = random.choice(unused)
            else:
                c = random.randint(0, 119)
        else:
            c = random.randint(0, 119)
            
        if c == colors[e]:
            continue
            
        # Check validity
        valid = True
        for c_idx in edge_to_cycles[e]:
            cycle_edges = cycles[c_idx]
            # Get colors of the 4 edges
            # One is e, which will be c
            # The others are colors[other_edge]
            col_set = set()
            for ed in cycle_edges:
                if ed == e:
                    col_set.add(c)
                else:
                    col_set.add(colors[ed])
            if len(col_set) == 4:
                valid = False
                break
                
        if valid:
            colors[e] = c
            used_colors.add(c)
            if len(used_colors) > best_num_colors:
                best_num_colors = len(used_colors)
                
    return colors
