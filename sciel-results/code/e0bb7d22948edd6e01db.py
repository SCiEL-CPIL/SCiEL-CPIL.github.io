import itertools
import time
import random

def construct():
    n = 16
    # Precompute edges and quads
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    # Map edge to index
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        
    quads = list(itertools.combinations(range(n), 4))
    
    # For each quad, find the 6 edge indices
    quad_edges = []
    for q in quads:
        a, b, c, d = q
        e1 = edge_index[(a, b)]
        e2 = edge_index[(a, c)]
        e3 = edge_index[(a, d)]
        e4 = edge_index[(b, c)]
        e5 = edge_index[(b, d)]
        e6 = edge_index[(c, d)]
        quad_edges.append([e1, e2, e3, e4, e5, e6])
        
    # For each edge, find the quads it belongs to
    num_quads = len(quads)
    edge_to_quads = [[] for _ in edges]
    for q_idx, e_list in enumerate(quad_edges):
        for e_idx in e_list:
            edge_to_quads[e_idx].append(q_idx)

    # Function to count distinct colors in a quad
    def count_distinct(e_list, colors):
        # e_list is list of 6 edge indices
        c1 = colors[e_list[0]]
        c2 = colors[e_list[1]]
        c3 = colors[e_list[2]]
        c4 = colors[e_list[3]]
        c5 = colors[e_list[4]]
        c6 = colors[e_list[5]]
        return len({c1, c2, c3, c4, c5, c6})

    # State: colors (list of 120 ints), bads (list of bools), total_bad (int)
    
    def local_search(m, time_limit):
        start_time = time.time()
        # Initialize colors
        # Try a few random restarts
        best_colors = None
        
        for restart in range(20): # 20 restarts
            colors = [random.randint(0, m-1) for _ in edges]
            
            # Calculate initial bads
            bads = [False] * num_quads
            total_bad = 0
            for q_idx, e_list in enumerate(quad_edges):
                if count_distinct(e_list, colors) < 5:
                    bads[q_idx] = True
                    total_bad += 1
                    
            if total_bad == 0:
                return colors

            # Coordinate descent
            while time.time() - start_time < time_limit:
                # Pick a random edge
                e_idx = random.randint(0, len(edges)-1)
                old_color = colors[e_idx]
                
                best_new_color = old_color
                min_bad = total_bad
                
                # Try all m colors
                for c in range(m):
                    if c == old_color:
                        continue
                    colors[e_idx] = c
                    
                    # Update bads for quads containing e_idx
                    delta = 0
                    for q_idx in edge_to_quads[e_idx]:
                        was_bad = bads[q_idx]
                        # Recalculate distinct
                        e_list = quad_edges[q_idx]
                        if count_distinct(e_list, colors) < 5:
                            is_bad = True
                        else:
                            is_bad = False
                            
                        if is_bad != was_bad:
                            bads[q_idx] = is_bad
                            if is_bad:
                                delta += 1
                            else:
                                delta -= 1
                                
                    new_total = total_bad + delta
                    
                    if new_total < min_bad:
                        min_bad = new_total
                        best_new_color = c
                        # Keep the updated bads state? 
                        # No, we need to revert if we don't accept.
                        # It's safer to revert bads if we don't accept.
                        # Or, we can just save the bads state if we accept.
                        # Let's revert bads for now to keep state consistent.
                        for q_idx in edge_to_quads[e_idx]:
                            # Revert to previous state? 
                            # This is tricky. We modified bads in place.
                            # We should save the old bads values.
                            pass
                            
                    # Revert colors
                    colors[e_idx] = old_color
                    # Revert bads? 
                    # The logic above is messy. Let's refactor.
                    
                # Refactored update logic:
                # 1. Save old bads for affected quads
                # 2. Try color, update bads, calculate total
                # 3. If better, accept (keep new bads, keep new color)
                # 4. Else, revert bads
                
                # Let's implement the refactored version inside the loop properly.
                
                # Actually, let's just implement a cleaner version below.
                break # Exit inner loop to implement cleaner version
                
            # End restarts
        return best_colors # Should not happen if we find one

    # Cleaner Local Search
    def run_local_search(m, time_budget):
        start_time = time.time()
        
        for restart in range(50): # More restarts
            colors = [random.randint(0, m-1) for _ in edges]
            
            bads = [False] * num_quads
            total_bad = 0
            for q_idx, e_list in enumerate(quad_edges):
                if count_distinct(e_list, colors) < 5:
                    bads[q_idx] = True
                    total_bad += 1
            
            if total_bad == 0:
                return colors

            # Coordinate Descent
            improved = True
            while improved and (time.time() - start_time < time_budget):
                improved = False
                # Shuffle edges for random order? Or just iterate.
                # Random order helps.
                edge_order = list(range(len(edges)))
                random.shuffle(edge_order)
                
                for e_idx in edge_order:
                    if time.time() - start_time > time_budget:
                        return None
                    old_color = colors[e_idx]
                    best_c = old_color
                    best_total = total_bad
                    best_bads_copy = None # Only save if we improve? No, we need to revert.
                    
                    # Save current bads state for affected quads
                    affected_quads = edge_to_quads[e_idx]
                    old_bads_state = {q_idx: bads[q_idx] for q_idx in affected_quads}
                    
                    for c in range(m):
                        if c == old_color:
                            continue
                        colors[e_idx] = c
                        
                        # Update bads
                        delta = 0
                        for q_idx in affected_quads:
                            was_bad = old_bads_state[q_idx] # Use original state for comparison? 
                            # No, if we change color, we recalc.
                            # But if we try c1, then c2, the state changes.
                            # We must revert to old_color state before trying next c.
                            # So, for each c:
                            # 1. Set color to c
                            # 2. Recalc bads for affected quads based on current colors
                            # 3. Calc total
                            # 4. Revert color to old_color? 
                            #    And revert bads to old_bads_state?
                            
                            # This is O(m * |affected|) recalculations.
                            # |affected| is ~91. m is ~13. 91*13 = 1183 ops per edge.
                            # 120 edges * 1183 = 141k ops per sweep.
                            # This is fine.
                            
                            pass
                        
                        # Implementation:
                        # We need to efficiently count distinct.
                        # Let's just do it.
                        
                        # Recalc bads for affected quads
                        new_bads = {}
                        new_total = total_bad
                        # We need to subtract old bads of affected quads first
                        for q_idx in affected_quads:
                            if old_bads_state[q_idx]:
                                new_total -= 1
                                
                        for q_idx in affected_quads:
                            e_list = quad_edges[q_idx]
                            if count_distinct(e_list, colors) < 5:
                                new_bads[q_idx] = True
                                new_total += 1
                            else:
                                new_bads[q_idx] = False
                                
                        if new_total < best_total:
                            best_total = new_total
                            best_c = c
                            best_bads_copy = new_bads
                            # Don't update global bads yet, just track best
                            
                        # Revert colors to old_color for next c trial
                        # (We don't need to revert bads because we didn't update them globally)
                        # We just used local variables.
                        
                    # End c loop
                    
                    if best_c != old_color:
                        # Accept move
                        colors[e_idx] = best_c
                        # Update global bads and total_bad
                        for q_idx in affected_quads:
                            if bads[q_idx] != best_bads_copy[q_idx]:
                                if bads[q_idx]:
                                    total_bad -= 1
                                if best_bads_copy[q_idx]:
                                    total_bad += 1
                                bads[q_idx] = best_bads_copy[q_idx]
                        total_bad = best_total
                        improved = True
                        if total_bad == 0:
                            return colors

            if total_bad == 0:
                return colors

        return None

    # Try m from 13 to 15
    for m in range(13, 16):
        # Allocate time
        # If 13 fails, try 14.
        # Give 30s to each? Total 90s.
        res = run_local_search(m, 30.0)
        if res:
            return res

    # Fallback to 17
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append((i + j) % 17)
    return colors

# EVOLVE-BLOCK-END
