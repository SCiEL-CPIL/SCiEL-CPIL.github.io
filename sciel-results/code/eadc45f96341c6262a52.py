import time
import itertools
import random
import numpy as np

def construct():
    n = 16
    deadline = time.time() + 140.0  # Safety margin
    
    # Precompute edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute K4s
    all_k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(all_k4s)
    
    # Map edge to index
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    
    # k4_edges[k4_idx] = list of 6 edge indices
    k4_edges = []
    for k4 in all_k4s:
        e_list = [edge_to_idx[(a, b)] for a, b in itertools.combinations(k4, 2)]
        k4_edges.append(e_list)
        
    # edge_k4s[edge_idx] = list of k4 indices containing this edge
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_k4s[e].append(k4_idx)

    # Fast validity check using sets
    def is_valid(colors):
        for e_list in k4_edges:
            if len(set(colors[e] for e in e_list)) < 5:
                return False
        return True
        
    def count_colors(colors):
        return len(set(colors))

    # Repair function: ensures validity by adding colors if necessary
    # We start with a random assignment of K colors.
    # If a K4 has < 5 colors, we pick an edge in it and assign it a color 
    # that is not currently in that K4 (if any exist in the palette) 
    # or a new color. To minimize colors, we prefer existing colors first.
    def repair(colors, max_colors):
        # max_colors is the current limit we want to try to stay within
        # If we exceed, we allow new colors but try to avoid it.
        # Actually, let's just use a large palette initially and eliminate later.
        # But the prompt says minimize. 
        # Strategy: Start with K random colors. If invalid, recolor an edge in a bad K4 
        # to a color not in that K4. If all K colors are in that K4, we must use a new color 
        # or change another edge in the K4. 
        # Simpler: Just ensure validity. Then eliminate.
        
        # We'll use a simple loop. If K4 is bad, pick an edge, try to change it to 
        # a color not in that K4.
        
        # To make this efficient, we track the current number of colors.
        # We allow up to, say, 20 colors during repair.
        max_allowed_colors = 20
        
        for _ in range(10000): # Max iterations
            # Find a bad K4
            bad_k4_idx = -1
            for k4_idx in range(num_k4s):
                e_list = k4_edges[k4_idx]
                if len(set(colors[e] for e in e_list)) < 5:
                    bad_k4_idx = k4_idx
                    break
            
            if bad_k4_idx == -1:
                return True # Valid
                
            e_list = k4_edges[bad_k4_idx]
            current_colors_in_k4 = set(colors[e] for e in e_list)
            
            # Pick a random edge from this K4
            e_idx = random.choice(e_list)
            
            # Find a valid new color
            # Prefer colors already in the global palette but not in this K4
            # If none, use a new color (if under limit)
            
            used_colors_global = set(colors)
            candidates = [c for c in used_colors_global if c not in current_colors_in_k4]
            
            if not candidates:
                # Need a new color
                if len(used_colors_global) < max_allowed_colors:
                    new_c = max(used_colors_global) + 1
                    candidates = [new_c]
                else:
                    # We are stuck with max colors, but K4 has all of them?
                    # If K4 has 5 distinct colors, it's valid.
                    # If it has < 5, and all global colors are in it, 
                    # it means the K4 uses only a subset of global colors?
                    # No, candidates are colors NOT in K4.
                    # If candidates is empty, it means all global colors ARE in K4.
                    # But K4 only has 6 edges. If global colors > 5, 
                    # K4 cannot contain all of them.
                    # So if global colors >= 5, there must be a color not in K4 
                    # unless K4 contains 5 distinct colors (which is valid).
                    # Wait, if K4 is invalid, it has < 5 distinct colors.
                    # So it uses at most 4 colors.
                    # If global palette has >= 5 colors, there is a color not in K4.
                    # So candidates should not be empty if we have >= 5 colors.
                    # If we have < 5 colors globally, we must add a new one.
                    new_c = max(used_colors_global) + 1
                    candidates = [new_c]
            
            new_c = random.choice(candidates)
            colors[e_idx] = new_c
            
        return is_valid(colors)

    # Eliminate colors: try to remove one color at a time
    def eliminate_colors(colors):
        # Keep track of best
        best_colors = colors[:]
        best_count = count_colors(colors)
        
        while time.time() < deadline:
            current_count = count_colors(colors)
            if current_count < best_count:
                best_colors = colors[:]
                best_count = current_count
                if best_count <= 13: # Theoretical lower bound
                    break
                    
            used_colors = set(colors)
            if not used_colors:
                break
            
            # Pick a color to eliminate (random or least frequent)
            # Least frequent is often easier to eliminate
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Sort by count
            sorted_colors = sorted(used_colors, key=lambda c: color_counts[c])
            
            eliminated = False
            # Try to eliminate the rarest color
            target_color = sorted_colors[0]
            
            edges_with_target = [idx for idx, c in enumerate(colors) if c == target_color]
            
            # Try to recolor all edges of target_color to other colors
            # We need to ensure validity after each recoloring
            # This is a constraint satisfaction problem.
            # Greedy approach:
            for e_idx in edges_with_target:
                # Find a valid new color
                valid_new_colors = []
                other_colors = [c for c in used_colors if c != target_color]
                
                for new_c in other_colors:
                    # Check if recoloring e_idx to new_c keeps all K4s containing e_idx valid
                    # We only need to check K4s containing e_idx
                    ok = True
                    for k4_idx in edge_k4s[e_idx]:
                        e_list = k4_edges[k4_idx]
                        # Construct the set of colors in this K4 with the change
                        # Optimization: precompute colors of other edges?
                        # Just do it simply for now
                        k4_colors = set()
                        for e in e_list:
                            if e == e_idx:
                                k4_colors.add(new_c)
                            else:
                                k4_colors.add(colors[e])
                        if len(k4_colors) < 5:
                            ok = False
                            break
                    if ok:
                        valid_new_colors.append(new_c)
                
                if valid_new_colors:
                    # Pick one, maybe random or one that appears most
                    # To help future eliminations, pick a common color
                    valid_new_colors.sort(key=lambda c: -color_counts.get(c, 0))
                    colors[e_idx] = valid_new_colors[0]
                else:
                    # Cannot recolor this edge to an existing color
                    # We might need to introduce a new color? No, we want to eliminate.
                    # If we can't recolor to existing, we fail to eliminate this color.
                    # Revert? No, we haven't changed it yet.
                    # So we just fail.
                    eliminated = False
                    break
            else:
                # All edges recolored successfully
                eliminated = True
                used_colors.discard(target_color)
                # Update color_counts? Not strictly necessary if we recompute next loop
                # But we should update global state implicitly via colors[]
                
            if not eliminated:
                # Try a random color instead of the rarest?
                # Or try a random edge recoloring to break symmetry?
                # Let's try a random target color
                target_color = random.choice(list(used_colors))
                edges_with_target = [idx for idx, c in enumerate(colors) if c == target_color]
                
                success = True
                for e_idx in edges_with_target:
                    valid_new_colors = []
                    other_colors = [c for c in used_colors if c != target_color]
                    for new_c in other_colors:
                        ok = True
                        for k4_idx in edge_k4s[e_idx]:
                            e_list = k4_edges[k4_idx]
                            k4_colors = set()
                            for e in e_list:
                                if e == e_idx:
                                    k4_colors.add(new_c)
                                else:
                                    k4_colors.add(colors[e])
                            if len(k4_colors) < 5:
                                ok = False
                                break
                        if ok:
                            valid_new_colors.append(new_c)
                    
                    if valid_new_colors:
                        colors[e_idx] = random.choice(valid_new_colors)
                    else:
                        success = False
                        break
                
                if success:
                    used_colors.discard(target_color)
                    eliminated = True
                else:
                    # Failed to eliminate any color in this round
                    # Break to avoid infinite loop if stuck
                    # We could try random restarts here
                    break
        
        return best_colors

    # Main loop: Multi-start
    best_overall = None
    best_count_overall = float('inf')
    
    # Try different initial numbers of colors
    for init_k in [13, 14, 15, 16]:
        if time.time() > deadline:
            break
            
        for attempt in range(10): # 10 attempts per k
            if time.time() > deadline:
                break
                
            # Random initialization
            colors = [random.randint(0, init_k - 1) for _ in range(num_edges)]
            
            # Repair
            if not repair(colors, init_k + 5): # Allow some extra during repair
                continue
                
            # Eliminate
            result = eliminate_colors(colors)
            
            cnt = count_colors(result)
            if cnt < best_count_overall:
                best_count_overall = cnt
                best_overall = result
                
            if best_count_overall <= 13:
                return best_overall

    # If no result found (shouldn't happen), return a valid one from repair
    # Fallback: just return a valid coloring, even if not minimal
    if best_overall is None:
        colors = [random.randint(0, 19) for _ in range(num_edges)]
        repair(colors, 20)
        best_overall = colors

    return best_overall
