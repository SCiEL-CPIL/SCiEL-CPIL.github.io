import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
                    cycles.append(e1)
                    e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
                    cycles.append(e2)
                    e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
                    cycles.append(e3)
    return cycles

def construct():
    deadline = time.time() + 55
    
    cycles = build_four_cycles()
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)
    
    # Precompute for each edge, the set of other edges in each cycle it belongs to
    # For constraint checking: if edge e is assigned color c, then in any 4-cycle containing e,
    # the other 3 edges must NOT all have distinct colors none of which is c
    # i.e., the other 3 edges must have at most 2 distinct colors, or one of them has color c
    
    # Greedy initialization: assign colors to maximize distinct colors
    colors = [0] * NUM_EDGES
    used_colors = set([0])
    
    # Process edges in order of constraint tightness (most constrained first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(cycles_by_edge[e]), reverse=True)
    
    for e in edge_order:
        # Try to find a new color that is valid
        found = False
        for c in range(1, NUM_EDGES):
            if c in used_colors:
                continue
            valid = True
            for cycle in cycles_by_edge[e]:
                other_colors = [colors[oe] for oe in cycle if oe != e]
                other_set = set(other_colors)
                if len(other_set) == 3 and c not in other_set:
                    valid = False
                    break
            if valid:
                colors[e] = c
                used_colors.add(c)
                found = True
                break
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search with improved constraint checking
    rng = random.Random(42)
    
    # Precompute: for each edge, list of (cycle_idx, position_in_cycle) for fast lookup
    # Actually, let's precompute for each edge e, for each cycle containing e,
    # the tuple of the other 3 edges
    edge_cycle_others = [[] for _ in range(NUM_EDGES)]
    for e in range(NUM_EDGES):
        for cycle in cycles_by_edge[e]:
            others = tuple(oe for oe in cycle if oe != e)
            edge_cycle_others[e].append(others)
    
    while time.time() < deadline:
        improved = False
        used_colors = set(colors)
        
        # Try each edge, try all colors to find one that increases color count
        for e in edge_order:
            if time.time() > deadline:
                break
            cur = colors[e]
            # Try new colors first (not in used_colors)
            for nc in range(NUM_EDGES):
                if nc == cur:
                    continue
                if nc in used_colors:
                    continue  # Focus on count-increasing moves first
                valid = True
                for others in edge_cycle_others[e]:
                    other_set = set(colors[oe] for oe in others)
                    if len(other_set) == 3 and nc not in other_set:
                        valid = False
                        break
                if valid:
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    break
        
        # Update best
        cur_count = len(set(colors))
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        if not improved:
            # Try swaps and recoloring with existing colors to enable future improvements
            found_any = False
            for e in edge_order:
                if time.time() > deadline:
                    break
                cur = colors[e]
                # Try recoloring with an existing color (might free up a slot)
                for nc in range(NUM_EDGES):
                    if nc == cur:
                        continue
                    if nc not in used_colors:
                        continue  # Already tried above
                    valid = True
                    for others in edge_cycle_others[e]:
                        other_set = set(colors[oe] for oe in others)
                        if len(other_set) == 3 and nc not in other_set:
                            valid = False
                            break
                    if valid:
                        # Check if this is useful: does it free up color cur?
                        # Count how many edges have color cur
                        cur_count = sum(1 for c in colors if c == cur)
                        if cur_count == 1:
                            # Recoloring frees up color cur
                            colors[e] = nc
                            used_colors.discard(cur)
                            if nc in used_colors:
                                pass
                            else:
                                used_colors.add(nc)
                            found_any = True
                            break
                if found_any:
                    break
            
            if not found_any:
                # Perturbation: random valid recoloring
                num_perturb = rng.randint(3, 8)
                for _ in range(num_perturb):
                    if time.time() > deadline:
                        break
                    e = rng.randint(0, NUM_EDGES - 1)
                    nc = rng.randint(0, NUM_EDGES - 1)
                    if nc == colors[e]:
                        continue
                    valid = True
                    for others in edge_cycle_others[e]:
                        other_set = set(colors[oe] for oe in others)
                        if len(other_set) == 3 and nc not in other_set:
                            valid = False
                            break
                    if valid:
                        colors[e] = nc
                        found_any = True
            
            # After perturbation, try to improve again
            if found_any:
                used_colors = set(colors)
                for e in edge_order:
                    if time.time() > deadline:
                        break
                    cur = colors[e]
                    for nc in range(NUM_EDGES):
                        if nc == cur:
                            continue
                        if nc in used_colors:
                            continue
                        valid = True
                        for others in edge_cycle_others[e]:
                            other_set = set(colors[oe] for oe in others)
                            if len(other_set) == 3 and nc not in other_set:
                                valid = False
                                break
                        if valid:
                            colors[e] = nc
                            used_colors.add(nc)
                            improved = True
                            break
                    if improved:
                        break
                
                cur_count = len(set(colors))
                if cur_count > best_count:
                    best_count = cur_count
                    best_colors = colors[:]
    
    return best_colors
