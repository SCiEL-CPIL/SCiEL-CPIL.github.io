import numpy as np
import random
import time

def construct():
    n = 16
    M = n * (n - 1) // 2  # 120
    
    # Precompute edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edges)
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            
    # Precompute 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
    # We need to check all 3 possible 4-cycles on any 4 vertices?
    # The task says: "all three cycles on every four-vertex subset are checked"
    # For 4 vertices {v0, v1, v2, v3}, the 3 Hamiltonian cycles (up to rotation/reflection) are:
    # 1. v0-v1-v2-v3-v0
    # 2. v0-v1-v3-v2-v0
    # 3. v0-v2-v1-v3-v0
    # Note: A "4-cycle" in graph theory usually refers to the cycle graph C4.
    # In K4, there are 3 distinct C4s.
    # We need to ensure NO 4-cycle has 4 distinct colors.
    
    cycles = []
    cycles_for_edge = [[] for _ in range(M)]
    
    # Generate all combinations of 4 vertices
    from itertools import combinations
    vertex_combos = list(combinations(range(n), 4))
    
    for combo in vertex_combos:
        v0, v1, v2, v3 = combo
        # The 3 distinct C4s on these 4 vertices
        # 1. (v0,v1), (v1,v2), (v2,v3), (v3,v0)
        # 2. (v0,v1), (v1,v3), (v3,v2), (v2,v0) -> Edges: (v0,v1), (v1,v3), (v2,v3), (v0,v2)
        # 3. (v0,v2), (v2,v1), (v1,v3), (v3,v0) -> Edges: (v0,v2), (v1,v2), (v1,v3), (v0,v3)
        
        # Let's define the 3 sets of edges explicitly
        # Cycle 1: v0-v1-v2-v3-v0
        e1 = edge_to_idx[(v0, v1)]
        e2 = edge_to_idx[(v1, v2)]
        e3 = edge_to_idx[(v2, v3)]
        e4 = edge_to_idx[(v3, v0)]
        c1 = [e1, e2, e3, e4]
        cycles.append(c1)
        for e in c1:
            cycles_for_edge[e].append(len(cycles)-1)
            
        # Cycle 2: v0-v1-v3-v2-v0
        e1 = edge_to_idx[(v0, v1)]
        e2 = edge_to_idx[(v1, v3)]
        e3 = edge_to_idx[(v2, v3)]
        e4 = edge_to_idx[(v0, v2)]
        c2 = [e1, e2, e3, e4]
        cycles.append(c2)
        for e in c2:
            cycles_for_edge[e].append(len(cycles)-1)
            
        # Cycle 3: v0-v2-v1-v3-v0
        e1 = edge_to_idx[(v0, v2)]
        e2 = edge_to_idx[(v1, v2)]
        e3 = edge_to_idx[(v1, v3)]
        e4 = edge_to_idx[(v0, v3)]
        c3 = [e1, e2, e3, e4]
        cycles.append(c3)
        for e in c3:
            cycles_for_edge[e].append(len(cycles)-1)

    # Initialization: min(i,j) strategy
    # This assigns color min(i,j) to edge (i,j).
    # Colors used: 0..14. Total 15 colors.
    # This is guaranteed valid: In any C4, let v_min be the vertex with smallest index.
    # The two edges incident to v_min in the cycle will have color v_min.
    # Thus, at least two edges share a color. No rainbow C4.
    
    best_colors = [0] * M
    for idx, (i, j) in enumerate(idx_to_edge):
        best_colors[idx] = min(i, j)
        
    # Count distinct colors
    def count_colors(colors):
        return len(set(colors))
        
    best_score = count_colors(best_colors)
    current_colors = best_colors[:]
    current_score = best_score
    
    # Frequency array for colors 0..119
    max_color_val = 119
    freq = [0] * (max_color_val + 1)
    for c in current_colors:
        freq[c] += 1
        
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    T = 5.0
    # Local Search / SA
    # Objective: Maximize distinct colors.
    # Constraint: No rainbow C4.
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, M - 1)
        old_c = current_colors[e]
        
        # Pick new color
        # Bias towards colors that are currently unused or rarely used?
        # Or just random 0..119
        # To increase diversity, try to pick a color not in the set of current colors if possible?
        # But we don't know if it's valid.
        
        # Strategy: 50% random from 0..119, 50% from existing colors (to mix)
        if random.random() < 0.5:
            nc = random.randint(0, max_color_val)
        else:
            # Pick from existing
            existing = [c for c in range(max_color_val + 1) if freq[c] > 0]
            nc = random.choice(existing)
            
        if nc == old_c:
            continue
            
        # Check Validity
        valid = True
        for c_idx in cycles_for_edge[e]:
            c = cycles[c_idx]
            # Get other 3 edges
            others = [x for x in c if x != e]
            c1 = current_colors[others[0]]
            c2 = current_colors[others[1]]
            c3 = current_colors[others[2]]
            
            # If the other 3 are distinct, nc MUST be one of them
            if c1 != c2 and c1 != c3 and c2 != c3:
                if nc != c1 and nc != c2 and nc != c3:
                    valid = False
                    break
            
        if not valid:
            continue
            
        # Calculate Score Change
        old_freq = freq[old_c]
        new_freq = freq[nc]
        
        delta = 0
        if old_freq == 1:
            delta -= 1
        if new_freq == 0:
            delta += 1
            
        # Acceptance
        if delta > 0:
            accept = True
        elif delta == 0:
            # Neutral move: accept with some probability to explore
            accept = random.random() < 0.1
        else:
            # delta < 0
            accept = random.random() < np.exp(delta / T)
            
        if accept:
            current_colors[e] = nc
            freq[old_c] -= 1
            freq[nc] += 1
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_colors = current_colors[:]
                
        # Cooling
        T *= 0.999
        if T < 0.01:
            T = 0.01
            
    return best_colors
