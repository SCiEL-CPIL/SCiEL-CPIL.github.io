import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n = 20
    colors_count = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Precompute all potential edges
    all_edges = []
    edge_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_map[(u, v)] = len(all_edges) - 1
    
    # Precompute 4-cycles
    # A 4-cycle is a set of 4 edges. We represent it by the 4 edge indices.
    cycles = []
    cycles_set = set()
    
    # Generate all 4-vertex subsets
    for combo in itertools.combinations(range(n), 4):
        u, v, w, x = combo
        # The 3 distinct 4-cycles on these vertices
        # 1. u-v-w-x-u
        e1 = edge_map[(min(u,v), max(u,v))]
        e2 = edge_map[(min(v,w), max(v,w))]
        e3 = edge_map[(min(w,x), max(w,x))]
        e4 = edge_map[(min(x,u), max(x,u))]
        cycle_tuple = tuple(sorted([e1, e2, e3, e4]))
        if cycle_tuple not in cycles_set:
            cycles_set.add(cycle_tuple)
            cycles.append(cycle_tuple)
            
        # 2. u-v-x-w-u
        e1 = edge_map[(min(u,v), max(u,v))]
        e2 = edge_map[(min(v,x), max(v,x))]
        e3 = edge_map[(min(x,w), max(x,w))]
        e4 = edge_map[(min(w,u), max(w,u))]
        cycle_tuple = tuple(sorted([e1, e2, e3, e4]))
        if cycle_tuple not in cycles_set:
            cycles_set.add(cycle_tuple)
            cycles.append(cycle_tuple)
            
        # 3. u-w-v-x-u
        e1 = edge_map[(min(u,w), max(u,w))]
        e2 = edge_map[(min(w,v), max(v,v))] # Wait, min(w,v) is same as (v,w)
        e2 = edge_map[(min(w,v), max(w,v))]
        e3 = edge_map[(min(v,x), max(v,x))]
        e4 = edge_map[(min(x,u), max(x,u))]
        cycle_tuple = tuple(sorted([e1, e2, e3, e4]))
        if cycle_tuple not in cycles_set:
            cycles_set.add(cycle_tuple)
            cycles.append(cycle_tuple)
            
    num_cycles = len(cycles)
    
    # Map edge index to list of cycle indices it participates in
    edge_to_cycles = [[] for _ in range(len(all_edges))]
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)
    
    # Greedy construction with local search repair
    # Strategy:
    # 1. Start with empty graph.
    # 2. Order edges by "difficulty" (number of 4-cycles they are part of). 
    #    Edges with fewer cycles are easier to add.
    #    Actually, adding edges creates cycles. 
    #    Let's try adding edges in a random order, but check validity.
    #    If adding an edge creates a violation, try to recolor existing edges to fix it.
    #    If that fails, skip the edge.
    
    best_result = []
    best_num_edges = 0
    
    # Run multiple restarts with different random seeds/orders
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
            
        # Randomize edge order
        edge_indices = list(range(len(all_edges)))
        random.shuffle(edge_indices)
        
        # Sort by "constraint weight"? 
        # Edges in the middle of the graph are in more cycles.
        # Let's try adding edges with fewer cycles first (peripheral edges)
        # Or just random.
        
        # State
        colours = [None] * len(all_edges)
        active_edges = set()
        
        # We will iterate through edges in a specific order.
        # Order: random shuffle, but maybe bias towards edges that form fewer new 4-cycles?
        # For simplicity, let's use a heuristic order: 
        # Calculate the number of 4-cycles an edge is in.
        # Prefer edges with fewer cycles? No, that might lead to a sparse graph.
        # Prefer edges that are "independent" in some sense?
        
        # Let's try a simple greedy: 
        # Pick an edge. Try to assign a color.
        # Check all 4-cycles involving this edge.
        # If a cycle becomes "closed" (all 4 edges present), check if colors are distinct.
        # If not, try to recolor one of the other 3 edges in that cycle.
        
        # To make this efficient, we maintain a list of "potential" 4-cycles.
        # A 4-cycle is "active" if 3 of its edges are present.
        # When we add the 4th edge, we must ensure the 4 colors are distinct.
        
        # Data structures for fast lookup:
        # edge_present: bool array
        # edge_color: array of colors (or -1)
        # cycles_by_edge: precomputed
        
        # We will use a local search approach on the set of edges included.
        # But simpler: Greedy addition.
        
        # Let's try adding edges in random order.
        # If adding edge E creates a conflict in any 4-cycle C where the other 3 edges are already present:
        # 1. Try to change color of E.
        # 2. If that fails, try to change color of one of the other 3 edges in C.
        # 3. If that fails, reject E.
        
        # Optimization: 
        # We need to know which 4-cycles are "ready to close" (have 3 edges).
        
        # Let's implement this.
        
        edge_present = [False] * len(all_edges)
        edge_color = [-1] * len(all_edges)
        
        # Count how many edges are present in each cycle
        cycle_edge_count = [0] * num_cycles
        # Store the present edges in each cycle? Or just check on the fly?
        # On the fly is O(4) per check.
        
        # To speed up, we can maintain a list of cycles that have exactly 3 edges present.
        # "Almost full" cycles.
        almost_full_cycles = set()
        
        # When adding an edge e:
        # 1. Mark e as present.
        # 2. For each cycle c in edge_to_cycles[e]:
        #    - Increment cycle_edge_count[c]
        #    - If count becomes 3, add c to almost_full_cycles
        #    - If count becomes 4, the cycle is full. Check validity.
        #       If invalid, we have a conflict.
        
        # If conflict:
        # We need to resolve it. 
        # Option A: Change color of e.
        # Option B: Change color of one of the other 3 edges in c.
        # Option C: Remove e (reject).
        
        # Let's try a limited local search on colors when a conflict arises.
        
        def try_add_edge(e_idx):
            # Check if adding this edge causes any immediate conflict
            # that cannot be resolved by recoloring e or its neighbors in the cycle.
            
            # Tentatively add
            edge_present[e_idx] = True
            edge_color[e_idx] = -1 # Temporarily uncolored
            
            # Update cycle counts
            affected_cycles = edge_to_cycles[e_idx]
            
            # Find cycles that become full
            full_cycles = []
            for c_idx in affected_cycles:
                cycle_edge_count[c_idx] += 1
                if cycle_edge_count[c_idx] == 4:
                    full_cycles.append(c_idx)
                elif cycle_edge_count[c_idx] == 3:
                    almost_full_cycles.add(c_idx)
            
            if not full_cycles:
                # No full cycles created. Just need to assign a color to e_idx
                # that doesn't conflict with... wait, if no full cycle, there is no 4-cycle constraint yet.
                # But we must assign a color. Any color is fine for now?
                # Yes, because no 4-cycle is closed.
                # However, we should pick a color that is "compatible" with future additions?
                # Greedy: pick a random color.
                edge_color[e_idx] = random.randint(0, colors_count - 1)
                return True
            
            # There are full cycles. We must assign colors to e_idx and potentially others
            # such that all full cycles involving e_idx are rainbow.
            
            # Let's collect all full cycles involving e_idx
            # Actually, only cycles in `full_cycles` are newly full.
            # But e_idx might be in other full cycles? No, if it wasn't full before, 
            # and we just added the 4th edge, it is full now.
            # Wait, if e_idx was already present? No, we are adding it.
            
            # So we need to satisfy constraints from `full_cycles`.
            # Let's try to assign a color to e_idx.
            # For each candidate color for e_idx:
            #   Check if all full_cycles involving e_idx become rainbow.
            #   If a cycle is not rainbow, try to recolor one of the other 3 edges.
            
            # This is a CSP.
            # Let's try a simple heuristic:
            # 1. Try all 20 colors for e_idx.
            # 2. For each color, check if it's valid for all full_cycles.
            #    Validity check: For each full cycle c, let other_edges be the 3 other edges.
            #    Let their colors be c1, c2, c3.
            #    We need {col(e), c1, c2, c3} to be distinct.
            #    This implies col(e) not in {c1, c2, c3} AND c1, c2, c3 are distinct.
            #    Note: c1, c2, c3 are already assigned. If they are not distinct, 
            #    then the cycle was already invalid? 
            #    Wait, if the cycle was not full before, we didn't check it.
            #    So it's possible that the other 3 edges have duplicate colors.
            #    In that case, we MUST recolor one of them.
            
            # Strategy:
            # For each full cycle c involving e_idx:
            #   If the other 3 edges have duplicate colors, we must fix that first.
            #   Then, we must ensure e_idx's color is distinct from them.
            
            # Let's implement a function to check if a set of 4 edges can be colored rainbow
            # given that 3 are fixed and 1 is variable (e_idx).
            
            def can_color_edge(e_idx, full_cycles_list):
                # Returns True if there exists a color for e_idx such that
                # all cycles in full_cycles_list are rainbow.
                # Note: This assumes the other edges in the cycles are already validly colored
                # among themselves (i.e., no duplicates within the 3 fixed edges).
                
                # First, check if the fixed parts of the cycles are valid.
                for c_idx in full_cycles_list:
                    cycle_edges = cycles[c_idx]
                    fixed_edges = [e for e in cycle_edges if e != e_idx]
                    fixed_colors = [edge_color[e] for e in fixed_edges]
                    if len(set(fixed_colors)) < 3:
                        # The fixed edges themselves have a conflict.
                        # We cannot fix this by changing e_idx's color.
                        # We would need to recolor one of the fixed edges.
                        return False # Signal that recoloring of other edges is needed
                
                # Now, find a color for e_idx that is not in any of the fixed color sets.
                forbidden = set()
                for c_idx in full_cycles_list:
                    cycle_edges = cycles[c_idx]
                    fixed_edges = [e for e in cycle_edges if e != e_idx]
                    fixed_colors = [edge_color[e] for e in fixed_edges]
                    forbidden.update(fixed_colors)
                
                # Find a color in 0..19 not in forbidden
                for c in range(colors_count):
                    if c not in forbidden:
                        return True # We found a valid color
                return False

            # We need to handle the case where fixed edges have conflicts.
            # If can_color_edge returns False due to fixed conflicts, we try to recolor one of the fixed edges.
            
            # Let's try a more robust approach:
            # 1. Identify all "problematic" cycles (full cycles involving e_idx).
            # 2. Try to assign a color to e_idx.
            # 3. If a cycle has duplicate colors among the other 3 edges, pick one of those edges and recolor it.
            
            # Let's just try all 20 colors for e_idx.
            found_valid = False
            for new_color in range(colors_count):
                edge_color[e_idx] = new_color
                
                # Check all full cycles involving e_idx
                valid = True
                for c_idx in full_cycles:
                    cycle_edges = cycles[c_idx]
                    colors_in_cycle = [edge_color[e] for e in cycle_edges]
                    if len(set(colors_in_cycle)) < 4:
                        valid = False
                        break
                
                if valid:
                    found_valid = True
                    break
                else:
                    # Try to fix by recoloring one of the other edges in the conflicting cycle?
                    # This is expensive. Let's just reject if no simple color works.
                    # Or, let's try to recolor one of the other edges.
                    
                    # Pick the first conflicting cycle
                    # Find a cycle that is invalid
                    for c_idx in full_cycles:
                        cycle_edges = cycles[c_idx]
                        colors_in_cycle = [edge_color[e] for e in cycle_edges]
                        if len(set(colors_in_cycle)) < 4:
                            # Find an edge to recolor
                            # Try recoloring each of the other 3 edges
                            for other_e in cycle_edges:
                                if other_e == e_idx:
                                    continue
                                old_color = edge_color[other_e]
                                # Try new colors for other_e
                                for new_c in range(colors_count):
                                    if new_c == old_color:
                                        continue
                                    edge_color[other_e] = new_c
                                    
                                    # Check if this fixes the cycle and doesn't break others
                                    # This is getting complicated. 
                                    # Let's simplify: If simple assignment fails, reject the edge.
                                    # This might lead to a sparse graph, but it's a start.
                                    
                                    # Revert
                                    edge_color[other_e] = old_color
                            break # Only try to fix the first conflict
                    
            if not found_valid:
                # Reject edge
                # Revert changes
                edge_present[e_idx] = False
                edge_color[e_idx] = -1
                for c_idx in affected_cycles:
                    cycle_edge_count[c_idx] -= 1
                    if cycle_edge_count[c_idx] == 3:
                        almost_full_cycles.discard(c_idx) # It's back to 3, so it's almost full again?
                        # Wait, if it was 4 and we remove 1, it becomes 3. So it IS almost full.
                        # But we didn't add it to almost_full_cycles in the revert?
                        # We should manage almost_full_cycles carefully.
                        # Actually, we don't strictly need almost_full_cycles for the logic above,
                        # we just check full_cycles.
                        # We can drop almost_full_cycles optimization for now.
                return False
            else:
                return True

        # Main loop for this restart
        # We process edges in the shuffled order
        for e_idx in edge_indices:
            if time.time() > deadline - 2.0:
                break
            
            if try_add_edge(e_idx):
                continue
            else:
                # Edge rejected
                pass
        
        # End of restart
        # Count edges
        num_active = sum(1 for e in range(len(all_edges)) if edge_present[e])
        
        if num_active > best_num_edges:
            best_num_edges = num_active
            # Construct result
            result = []
            for e_idx in range(len(all_edges)):
                if edge_present[e_idx]:
                    u, v = all_edges[e_idx]
                    result.append([u, v, edge_color[e_idx]])
            best_result = result
            
            # Early stop if we get a very dense graph?
            # Max possible is 190.
            # If we get > 100, that's good.
            if best_num_edges > 100:
                break

    # Fallback
    if not best_result:
        # Star
        return [[0, v, 0] for v in range(1, 20)]
    
    return best_result

# EVOLVE-BLOCK-END
