<<<<<<< SEARCH
    # Try multiple initial seeds and search strategies
    num_restarts = 8
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * time_per_restart
        if time.time() > restart_deadline:
            break
        
        # Different initial colorings
        if restart == 0:
            # Standard seed: (u + v) % 17
            colors = [0] * num_edges
            for i, (u, v) in enumerate(edges):
                colors[i] = (u + v) % 17
        elif restart == 1:
            # Alternative seed: (u * 3 + v * 5) % 17
            colors = [0] * num_edges
            for i, (u, v) in enumerate(edges):
                colors[i] = (u * 3 + v * 5) % 17
        elif restart == 2:
            # Random valid coloring attempt
            colors = [0] * num_edges
            for i, (u, v) in enumerate(edges):
                colors[i] = random.randint(0, 16)
            # Fix invalid K4s by adjusting colors
            for _ in range(100):
                valid = True
                for k4_idx in range(num_k4s):
                    e_list = k4_edges[k4_idx]
                    if len({colors[e] for e in e_list}) < 5:
                        valid = False
                        # Find an edge in this K4 that can be changed
                        for e in e_list:
                            for c in range(17):
                                if c != colors[e] and can_change_edge(e, c, colors):
                                    colors[e] = c
                                    break
                            if len({colors[x] for x in e_list}) >= 5:
                                break
                if valid:
                    break
        else:
            # Perturb the best found so far
            if best_colors is None:
                colors = [0] * num_edges
                for i, (u, v) in enumerate(edges):
                    colors[i] = (u + v) % 17
            else:
                colors = best_colors[:]
                # Randomly perturb some edges
                for _ in range(20):
                    e_idx = random.randint(0, num_edges - 1)
                    for c in range(17):
                        if c != colors[e_idx] and can_change_edge(e_idx, c, colors):
                            colors[e_idx] = c
                            break
        
        # Verify initial coloring is valid
        if not is_valid_coloring(colors):
            continue
        
        current_colors = colors[:]
        current_num_colors = count_colors(current_colors)
        
        if current_num_colors < best_num_colors:
            best_num_colors = current_num_colors
            best_colors = current_colors[:]
        
        # Local search
        iter_count = 0
        no_improve_count = 0
        
        while time.time() < restart_deadline:
            # Try to improve
            improved = False
            
            # Try a batch of random edges
            for _ in range(500):
                e_idx = random.randint(0, num_edges - 1)
                current_c = current_colors[e_idx]
                
                # Try to lower
                min_valid = -1
                for c in range(0, current_c):
                    if can_change_edge(e_idx, c, current_colors):
                        min_valid = c
                        break
                
                if min_valid != -1:
                    current_colors[e_idx] = min_valid
                    improved = True
            
            # Check for improvement in color count
            new_num_colors = count_colors(current_colors)
            if new_num_colors < best_num_colors:
                best_num_colors = new_num_colors
                best_colors = current_colors[:]
                no_improve_count = 0
            elif new_num_colors < current_num_colors:
                current_num_colors = new_num_colors
                no_improve_count = 0
            elif improved:
                # Colors changed but count didn't decrease
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            # If stuck, try a kick
            if no_improve_count > 50:
                # Try to change a random edge to a different valid color
                e_idx = random.randint(0, num_edges - 1)
                current_c = current_colors[e_idx]
                
                found_kick = False
                # Try colors in a wider range
                max_kick_color = max(current_colors) + 10
                for _ in range(30):
                    new_c = random.randint(0, max_kick_color)
                    if new_c == current_c:
                        continue
                    if can_change_edge(e_idx, new_c, current_colors):
                        current_colors[e_idx] = new_c
                        found_kick = True
                        break
                
                if not found_kick:
                    # Try to increase a color to break symmetry
                    e_idx = random.randint(0, num_edges - 1)
                    current_c = current_colors[e_idx]
                    for new_c in range(current_c + 1, current_c + 20):
                        if can_change_edge(e_idx, new_c, current_colors):
                            current_colors[e_idx] = new_c
                            found_kick = True
                            break
                
                no_improve_count = 0
            
            # Periodic compaction
            if iter_count % 200 == 0:
                compacted = compact_colors(current_colors)
                compacted_count = count_colors(compacted)
                if compacted_count < best_num_colors:
                    best_num_colors = compacted_count
                    best_colors = compacted[:]
                current_colors = compacted
                current_num_colors = compacted_count
            
            iter_count += 1
=======
    # Try to reduce from 15 to 14 or 13 colors using local search
    # Focus on merging color classes
    
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * time_per_restart
        if time.time() > restart_deadline:
            break
        
        # Start from the 1-factorization or perturb it
        if restart == 0:
            # Use the 1-factorization directly
            colors = colors_1f[:]
        elif restart == 1:
            # Try to merge two color classes from the 1-factorization
            colors = colors_1f[:]
            # Pick two color classes to try to merge
            c1, c2 = random.sample(range(15), 2)
            # Try to recolor edges of color c2 to color c1 where valid
            for e_idx in range(num_edges):
                if colors[e_idx] == c2:
                    if can_change_edge(e_idx, c1, colors):
                        colors[e_idx] = c1
        elif restart == 2:
            # Another merge attempt
            colors = colors_1f[:]
            c1, c2 = random.sample(range(15), 2)
            # Try to recolor edges of color c1 to color c2
            for e_idx in range(num_edges):
                if colors[e_idx] == c1:
                    if can_change_edge(e_idx, c2, colors):
                        colors[e_idx] = c2
        else:
            # Perturb the best found so far
            colors = best_colors[:]
            # Randomly perturb some edges
            for _ in range(30):
                e_idx = random.randint(0, num_edges - 1)
                for c in range(15):
                    if c != colors[e_idx] and can_change_edge(e_idx, c, colors):
                        colors[e_idx] = c
                        break
        
        # Verify initial coloring is valid
        if not is_valid_coloring(colors):
            continue
        
        current_colors = colors[:]
        current_num_colors = count_colors(current_colors)
        
        if current_num_colors < best_num_colors:
            best_num_colors = current_num_colors
            best_colors = current_colors[:]
        
        # Local search: try to reduce color count
        iter_count = 0
        no_improve_count = 0
        
        while time.time() < restart_deadline:
            # Try to improve
            improved = False
            
            # Strategy 1: Try to recolor an edge to a lower color
            for _ in range(300):
                e_idx = random.randint(0, num_edges - 1)
                current_c = current_colors[e_idx]
                
                min_valid = -1
                for c in range(0, current_c):
                    if can_change_edge(e_idx, c, current_colors):
                        min_valid = c
                        break
                
                if min_valid != -1:
                    current_colors[e_idx] = min_valid
                    improved = True
            
            # Strategy 2: Try to merge color classes
            # Find the color with the fewest edges and try to merge it
            color_counts = {}
            for c in current_colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            if color_counts:
                # Pick a color class to try to eliminate
                target_color = min(color_counts.keys(), key=lambda c: color_counts[c])
                target_count = color_counts[target_color]
                
                if target_count <= 8:  # Only try to merge small color classes
                    # Try to recolor all edges of target_color to other colors
                    edges_of_target = [e for e in range(num_edges) if current_colors[e] == target_color]
                    
                    for e_idx in edges_of_target:
                        # Try to recolor to a different color
                        for c in range(15):
                            if c != target_color and can_change_edge(e_idx, c, current_colors):
                                current_colors[e_idx] = c
                                improved = True
                                break
            
            # Check for improvement in color count
            new_num_colors = count_colors(current_colors)
            if new_num_colors < best_num_colors:
                best_num_colors = new_num_colors
                best_colors = current_colors[:]
                no_improve_count = 0
            elif new_num_colors < current_num_colors:
                current_num_colors = new_num_colors
                no_improve_count = 0
            elif improved:
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            # If stuck, try a kick
            if no_improve_count > 30:
                # Try to change a random edge to a different valid color
                e_idx = random.randint(0, num_edges - 1)
                current_c = current_colors[e_idx]
                
                found_kick = False
                for _ in range(20):
                    new_c = random.randint(0, 14)
                    if new_c == current_c:
                        continue
                    if can_change_edge(e_idx, new_c, current_colors):
                        current_colors[e_idx] = new_c
                        found_kick = True
                        break
                
                no_improve_count = 0
            
            # Periodic compaction
            if iter_count % 50 == 0:
                compacted = compact_colors(current_colors)
                compacted_count = count_colors(compacted)
                if compacted_count < best_num_colors:
                    best_num_colors = compacted_count
                    best_colors = compacted[:]
                current_colors = compacted
                current_num_colors = compacted_count
            
            iter_count += 1
>>>>>>> REPLACE

