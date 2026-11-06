import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Precompute K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        e_indices = []
        for a, b in itertools.combinations(quad, 2):
            e_indices.append(edge_to_idx[(min(a,b), max(a,b))])
        k4s.append(e_indices)
        
    # Convert to numpy arrays for faster access if needed, but lists are okay for 1820 items
    # We will use a fast validation function
    
    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for k4 in k4s:
            # Unpack for speed
            c0 = colors[k4[0]]
            c1 = colors[k4[1]]
            c2 = colors[k4[2]]
            c3 = colors[k4[3]]
            c4 = colors[k4[4]]
            c5 = colors[k4[5]]
            
            # Count distinct colors without set creation overhead if possible?
            # For 6 items, set is usually fast enough in CPython, but let's try a manual check
            # Or just use set, it's 1820 * 6 ops.
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # 1. Generate 1-factorization of K16
    # Vertex 15 is center. Vertices 0..14 on cycle.
    # Factor i (0..14):
    #   Edge (15, i)
    #   Edges (i+j mod 15, i-j mod 15) for j=1..7
    
    initial_colors = [0] * num_edges
    
    for i in range(15):
        # Edge (15, i)
        idx = edge_to_idx[(15, i)]
        initial_colors[idx] = i
        
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            idx = edge_to_idx[(a, b)]
            initial_colors[idx] = i
            
    current_colors = initial_colors[:]
    current_count = count_colors(current_colors)
    
    # Check if initial 1-factorization is valid
    if not is_valid(current_colors):
        # Repair phase: Try to fix violations
        # We will iterate: find a bad K4, pick an edge in it, try to recolor it.
        # To avoid infinite loops, we limit iterations or use randomization.
        
        max_repair_steps = 5000
        steps = 0
        while not is_valid(current_colors) and steps < max_repair_steps and time.time() < deadline:
            steps += 1
            
            # Find a violating K4
            bad_k4 = None
            for k4 in k4s:
                c0 = current_colors[k4[0]]
                c1 = current_colors[k4[1]]
                c2 = current_colors[k4[2]]
                c3 = current_colors[k4[3]]
                c4 = current_colors[k4[4]]
                c5 = current_colors[k4[5]]
                if len({c0, c1, c2, c3, c4, c5}) < 5:
                    bad_k4 = k4
                    break
            
            if bad_k4 is None:
                break # Should not happen if is_valid is consistent
                
            # Try to recolor one of the edges in this K4
            # Randomly pick an edge index from the K4
            edge_idx = random.choice(bad_k4)
            old_color = current_colors[edge_idx]
            
            # Try assigning a new color
            # Candidate colors: existing colors + new color (if we want to increase)
            # But we want to minimize, so try existing colors first.
            # If we can't find a valid existing color, we might need a new one, 
            # but that increases count. 
            # Actually, if we are stuck, maybe we should just accept a higher count 
            # and try to merge later? 
            # Or, we can try to swap colors between two edges?
            
            # Let's try recoloring to a random existing color first.
            existing_colors = list(set(current_colors))
            random.shuffle(existing_colors)
            
            success = False
            for new_color in existing_colors:
                if new_color == old_color:
                    continue
                # Apply change
                current_colors[edge_idx] = new_color
                if is_valid(current_colors):
                    success = True
                    break
                else:
                    # Revert
                    current_colors[edge_idx] = old_color
            
            if not success:
                # Try a new color? This increases count.
                # If we do this, we might end up with > 15 colors.
                # But if we can't fix it with existing, we have no choice.
                # However, 1-factorization is "close" to valid.
                # Let's try swapping colors of two edges in the K4?
                # Or just give up repair and fall back to (i+j)%17?
                # Let's try one random recolor to a new color to escape local optima
                new_color = max(existing_colors) + 1
                current_colors[edge_idx] = new_color
                # We don't check validity here, the next loop will handle it.
                # This might oscillate.
                # Better: if we can't fix, revert and try a different edge in the same K4.
                current_colors[edge_idx] = old_color
                
                # If we tried all edges in this K4 and failed, we are stuck.
                # Let's just break and use the fallback for now, 
                # or try a random perturbation.
                # For robustness, if repair fails, we fall back to the known valid 17-coloring.
                break

    if not is_valid(current_colors):
        # Fallback to known valid 17-coloring
        current_colors = [(i + j) % 17 for i, j in edges]
        current_count = count_colors(current_colors)

    # 2. Local Search: Merging colors
    # Now we have a valid coloring. Try to reduce color count.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current_colors)
        color_list = list(current_set)
        random.shuffle(color_list)
        
        for c1 in color_list:
            for c2 in color_list:
                if c1 == c2:
                    continue
                # Try merging c1 into c2
                # Optimization: Only change edges with color c1
                edges_to_change = [i for i in range(num_edges) if current_colors[i] == c1]
                
                if not edges_to_change:
                    continue
                    
                # Apply merge
                for i in edges_to_change:
                    current_colors[i] = c2
                    
                if is_valid(current_colors):
                    current_count = count_colors(current_colors)
                    improved = True
                    # Restart loop with new color set
                    break
                else:
                    # Revert
                    for i in edges_to_change:
                        current_colors[i] = c1
            
            if improved:
                break

    return current_colors

# EVOLVE-BLOCK-END
