import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = []
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
    
    def can_change_edge(e_idx, new_color, colors):
        old_color = colors[e_idx]
        if old_color == new_color:
            return True
        
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            other_colors = set()
            for e in e_list:
                if e != e_idx:
                    other_colors.add(colors[e])
            
            if new_color in other_colors:
                if len(other_colors) < 5:
                    return False
            else:
                if len(other_colors) < 4:
                    return False
        return True
    
    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid_coloring(colors):
        for k4_idx in range(num_k4s):
            e_list = k4_edges[k4_idx]
            if len({colors[e] for e in e_list}) < 5:
                return False
        return True
    
    best_colors = None
    best_num_colors = 17  # Start with the seed's color count
    
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
    
    # Final polish: try to reduce colors further from the best found
    if best_colors is not None:
        final_colors = best_colors[:]
        final_count = count_colors(final_colors)
        
        # Try greedy reduction one more time
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            e_idx = random.randint(0, num_edges - 1)
            current_c = final_colors[e_idx]
            
            min_valid = -1
            for c in range(0, current_c):
                if can_change_edge(e_idx, c, final_colors):
                    min_valid = c
                    break
            
            if min_valid != -1:
                final_colors[e_idx] = min_valid
        
        compacted = compact_colors(final_colors)
        if count_colors(compacted) < best_num_colors:
            best_colors = compacted
            best_num_colors = count_colors(compacted)
    
    # Return the best coloring found
    if best_colors is None:
        # Fallback: return the standard seed
        colors = [0] * num_edges
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 17
        return compact_colors(colors)
    
    return compact_colors(best_colors)
