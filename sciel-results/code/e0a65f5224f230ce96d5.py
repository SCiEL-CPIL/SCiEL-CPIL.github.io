import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115  # 5s margin
    
    # Precompute all edges
    edges = []
    edge_id = {}
    for u in range(n):
        for v in range(u+1, n):
            edge_id[(u, v)] = len(edges)
            edges.append((u, v))
    
    m_max = len(edges)
    
    # Precompute all 4-cycles
    cycles = []
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Cycle 1: a-b-c-d-a
                    e1 = edge_id.get((a,b))
                    e2 = edge_id.get((b,c))
                    e3 = edge_id.get((c,d))
                    e4 = edge_id.get((d,a))
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        cycles.append((e1, e2, e3, e4))
                        
                    # Cycle 2: a-b-d-c-a
                    e1 = edge_id.get((a,b))
                    e2 = edge_id.get((b,d))
                    e3 = edge_id.get((d,c))
                    e4 = edge_id.get((c,a))
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        cycles.append((e1, e2, e3, e4))
                        
                    # Cycle 3: a-c-b-d-a
                    e1 = edge_id.get((a,c))
                    e2 = edge_id.get((c,b))
                    e3 = edge_id.get((b,d))
                    e4 = edge_id.get((d,a))
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        cycles.append((e1, e2, e3, e4))
    
    num_cycles = len(cycles)
    
    # Map edge index to list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(m_max)]
    for i, cyc in enumerate(cycles):
        for e_idx in cyc:
            edge_to_cycles[e_idx].append(i)
            
    # State
    active = [True] * m_max
    colors = [0] * m_max
    cycle_edge_count = [4] * num_cycles # Initially all active
    
    # Initialize colors randomly
    random.seed(42)
    for i in range(m_max):
        colors[i] = random.randint(0, n-1)
        
    # cycle_is_bad: 1 if not rainbow, 0 if rainbow
    cycle_is_bad = [0] * num_cycles
    edge_badness = [0] * m_max
    
    # Initial calculation
    for i in range(num_cycles):
        if not (len(set([colors[cycles[i][0]], colors[cycles[i][1]], colors[cycles[i][2]], colors[cycles[i][3]]])) == 4):
            cycle_is_bad[i] = 1
            for e_idx in cycles[i]:
                edge_badness[e_idx] += 1
                
    total_bad = sum(cycle_is_bad)
    
    best_edges_found = 0
    best_active_copy = None
    best_colors_copy = None
    
    while time.time() < deadline:
        if total_bad == 0:
            current_active_count = sum(active)
            if current_active_count > best_edges_found:
                best_edges_found = current_active_count
                best_active_copy = active.copy()
                best_colors_copy = colors.copy()
            # If we found a valid graph, we can try to restart to see if we can get a denser one?
            # No, we started with max density. Any removal reduces density.
            # So the first valid graph found is the best for this run.
            # We can break and return.
            break
            
        # Find edge with max badness
        max_bad = -1
        target_edge = -1
        for e in range(m_max):
            if active[e]:
                if edge_badness[e] > max_bad:
                    max_bad = edge_badness[e]
                    target_edge = e
                    
        if target_edge == -1:
            break
            
        # Try to recolor target_edge
        best_new_badness = max_bad
        best_color = colors[target_edge]
        
        for new_c in range(n):
            if new_c == colors[target_edge]:
                continue
                
            delta = 0
            for cyc_idx in edge_to_cycles[target_edge]:
                if cycle_edge_count[cyc_idx] < 4:
                    continue
                
                cyc = cycles[cyc_idx]
                other_edges = [e for e in cyc if e != target_edge]
                c_others = [colors[e] for e in other_edges]
                
                current_colors = c_others + [colors[target_edge]]
                was_bad = (len(set(current_colors)) < 4)
                
                new_colors = c_others + [new_c]
                is_bad = (len(set(new_colors)) < 4)
                
                if was_bad and not is_bad:
                    delta -= 1
                elif not was_bad and is_bad:
                    delta += 1
                    
            new_total_badness = max_bad + delta
            
            if new_total_badness < best_new_badness:
                best_new_badness = new_total_badness
                best_color = new_c
                
        if best_new_badness < max_bad:
            # Apply recolor
            old_c = colors[target_edge]
            colors[target_edge] = best_color
            
            for cyc_idx in edge_to_cycles[target_edge]:
                if cycle_edge_count[cyc_idx] < 4:
                    continue
                
                cyc = cycles[cyc_idx]
                other_edges = [e for e in cyc if e != target_edge]
                c_others = [colors[e] for e in other_edges]
                
                old_colors = c_others + [old_c]
                was_bad = (len(set(old_colors)) < 4)
                
                new_colors = c_others + [best_color]
                is_bad = (len(set(new_colors)) < 4)
                
                if was_bad and not is_bad:
                    cycle_is_bad[cyc_idx] = 0
                    total_bad -= 1
                    for e_idx in cyc:
                        edge_badness[e_idx] -= 1
                elif not was_bad and is_bad:
                    cycle_is_bad[cyc_idx] = 1
                    total_bad += 1
                    for e_idx in cyc:
                        edge_badness[e_idx] += 1
        else:
            # Remove the edge
            active[target_edge] = False
            
            for cyc_idx in edge_to_cycles[target_edge]:
                cycle_edge_count[cyc_idx] -= 1
                if cycle_edge_count[cyc_idx] < 4:
                    # Cycle is no longer a constraint
                    if cycle_is_bad[cyc_idx]:
                        total_bad -= 1
                        cycle_is_bad[cyc_idx] = 0
                        for e_idx in cycles[cyc_idx]:
                            edge_badness[e_idx] -= 1
                # If it was already good, no change to total_bad, but edge_badness for other edges in cycle?
                # No, if cycle is inactive, it doesn't contribute to badness.
                # If it was active and good, it didn't contribute.
                # If it was active and bad, we just handled it.
                
    res = []
    if best_active_copy:
        for i in range(m_max):
            if best_active_copy[i]:
                u, v = edges[i]
                res.append([u, v, best_colors_copy[i]])
    else:
        # Fallback: Return a known valid small graph (e.g., a tree)
        for v in range(1, n):
            res.append([0, v, 0])
            
    return res
