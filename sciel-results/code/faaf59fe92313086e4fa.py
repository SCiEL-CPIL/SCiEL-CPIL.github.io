import time
import random
import itertools
import numpy as np

def construct():
    n = 16
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    # Precompute all K4s and their edge indices
    # Edge index for (i,j) with i<j: sum of sizes of previous i-rows + (j - i - 1)
    # Actually, let's just enumerate
    edge_list = []
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            edge_list.append((i,j))
            idx += 1
    # idx should be 120
    
    # Precompute K4s as lists of edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        edge_indices = [edge_idx[e] for e in edges]
        k4s.append(edge_indices)
    
    num_k4s = len(k4s)  # 1820
    
    # Build: for each edge, which K4s does it belong to?
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_indices in enumerate(k4s):
        for ei in edge_indices:
            edge_to_k4s[ei].append(k4_idx)
    
    def count_violations(colors):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for edge_indices in k4s:
            c_set = set()
            for ei in edge_indices:
                c_set.add(colors[ei])
            if len(c_set) < 5:
                violations += 1
        return violations
    
    def get_num_colours(colors):
        return len(set(colors))
    
    # Generate 1-factorization of K16
    # Standard construction: vertices 0..14 and infinity
    # For K_{2m}, 1-factorization: for round r in 0..2m-2, 
    #   matching r: pair infinity with vertex r, 
    #   and for k in 1..m-1: pair (r+k) mod (2m-1) with (r-k) mod (2m-1)
    def generate_1_factorization():
        m = n // 2  # 8
        # 2m-1 = 15 vertices numbered 0..14, plus infinity = 15
        # Actually let's use 0..14 for the 15 vertices, and 15 for infinity
        # Wait, n=16, so 2m=16, m=8, 2m-1=15
        # Vertices: 0..14 (15 vertices) and 15 (infinity)
        factorization = []
        for r in range(15):  # 15 rounds
            matching = []
            # Pair infinity (15) with vertex r
            matching.append((r, 15))
            # For k in 1..m-1 = 1..7:
            for k in range(1, 8):
                a = (r + k) % 15
                b = (r - k) % 15
                if a > b:
                    a, b = b, a
                matching.append((a, b))
            factorization.append(matching)
        return factorization
    
    factorization = generate_1_factorization()
    
    # Create initial coloring: edge color = matching index (0..14)
    colors = [0] * 120
    for match_idx, matching in enumerate(factorization):
        for (u, v) in matching:
            if u > v:
                u, v = v, u
            ei = edge_idx[(u, v)]
            colors[ei] = match_idx
    
    # Verify initial coloring is valid
    initial_violations = count_violations(colors)
    initial_num_colours = get_num_colours(colors)
    
    # Local search: try to change individual edge colors to reduce total distinct colors
    # while maintaining validity (0 violations)
    
    best_colors = colors[:]
    best_num_colours = initial_num_colours
    current_colors = colors[:]
    current_violations = initial_violations
    
    # We want to minimize num_colours subject to violations == 0
    # Strategy: for each edge, try changing its color to a color already in use
    # (or to a new color if it helps reduce violations)
    
    # Track which colors are in use
    def try_improve(current_colors, current_violations, max_iters):
        """Local search to minimize num_colours."""
        best_local = current_colors[:]
        best_local_viol = current_violations
        best_local_num = get_num_colours(current_colors)
        
        colors_arr = current_colors[:]
        viol = current_violations
        num_colours = get_num_colours(colors_arr)
        
        # Track color counts for quick updates
        color_counts = {}
        for c in colors_arr:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # For each K4, track the set of colors
        k4_color_sets = []
        for edge_indices in k4s:
            s = set()
            for ei in edge_indices:
                s.add(colors_arr[ei])
            k4_color_sets.append(s)
        
        iter_count = 0
        no_improve = 0
        while iter_count < max_iters:
            if time.time() - start_time > timeout - 5:
                break
            
            # Pick a random edge
            ei = random.randint(0, 119)
            old_color = colors_arr[ei]
            
            # Try changing to a different color
            # Candidate colors: colors currently in use, plus maybe a few new ones
            candidate_colors = list(color_counts.keys())
            if len(candidate_colors) < 15:
                # Add some new color options
                for c in range(15, 20):
                    if c not in color_counts:
                        candidate_colors.append(c)
            
            random.shuffle(candidate_colors)
            
            improved = False
            for new_color in candidate_colors:
                if new_color == old_color:
                    continue
                
                # Apply change
                colors_arr[ei] = new_color
                
                # Update color counts
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    del color_counts[old_color]
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
                
                # Update affected K4s
                affected_k4s = edge_to_k4s[ei]
                valid_change = True
                for k4_idx in affected_k4s:
                    k4_color_sets[k4_idx].discard(old_color)
                    k4_color_sets[k4_idx].add(new_color)
                    if len(k4_color_sets[k4_idx]) < 5:
                        valid_change = False
                        break
                
                if valid_change:
                    new_num = len(color_counts)
                    if new_num < num_colours:
                        num_colours = new_num
                        if num_colours < best_local_num:
                            best_local = colors_arr[:]
                            best_local_viol = viol
                            best_local_num = num_colours
                            no_improve = 0
                        else:
                            no_improve += 1
                        improved = True
                        break
                    elif new_num == num_colours and random.random() < 0.3:
                        # Accept with some probability for diversity
                        no_improve += 1
                        improved = True
                        break
                else:
                    # Revert
                    colors_arr[ei] = old_color
                    color_counts[old_color] = color_counts.get(old_color, 0) + 1
                    color_counts[new_color] -= 1
                    if color_counts[new_color] == 0:
                        del color_counts[new_color]
                    for k4_idx in affected_k4s:
                        k4_color_sets[k4_idx].discard(new_color)
                        k4_color_sets[k4_idx].add(old_color)
            
            if not improved:
                no_improve += 1
            
            iter_count += 1
            if no_improve > 500:
                break
        
        return best_local, best_local_viol, best_local_num
    
    # Run multiple restarts
    for restart in range(20):
        if time.time() - start_time > timeout - 10:
            break
        
        # Perturb the current best
        if restart == 0:
            current_colors = best_colors[:]
            current_violations = count_violations(current_colors)
        else:
            # Random perturbation
            current_colors = best_colors[:]
            for _ in range(random.randint(5, 20)):
                ei = random.randint(0, 119)
                current_colors[ei] = random.randint(0, 14)
            current_violations = count_violations(current_colors)
        
        # If we have violations, first try to fix them
        if current_violations > 0:
            # Try to fix violations by changing colors
            for fix_iter in range(2000):
                if time.time() - start_time > timeout - 5:
                    break
                if current_violations == 0:
                    break
                
                # Find a violating K4
                violating_k4 = None
                for k4_idx, edge_indices in enumerate(k4s):
                    c_set = set()
                    for ei in edge_indices:
                        c_set.add(current_colors[ei])
                    if len(c_set) < 5:
                        violating_k4 = k4_idx
                        break
                
                if violating_k4 is None:
                    current_violations = 0
                    break
                
                # Try to fix by changing one of the edges in this K4
                edge_indices = k4s[violating_k4]
                random.shuffle(edge_indices)
                fixed = False
                for ei in edge_indices:
                    old_c = current_colors[ei]
                    # Try new colors
                    for new_c in range(15):
                        if new_c == old_c:
                            continue
                        current_colors[ei] = new_c
                        new_viol = count_violations(current_colors)
                        if new_viol < current_violations:
                            current_violations = new_viol
                            fixed = True
                            break
                        else:
                            current_colors[ei] = old_c
                    if fixed:
                        break
        
        # Now do local search to minimize colors
        if current_violations == 0:
            improved_colors, improved_viol, improved_num = try_improve(
                current_colors, current_violations, max_iters=3000
            )
            if improved_viol == 0 and improved_num < best_num_colours:
                best_colors = improved_colors
                best_num_colours = improved_num
    
    # Final verification
    if count_violations(best_colors) != 0:
        # Fallback: use the 1-factorization coloring (15 colors)
        for match_idx, matching in enumerate(factorization):
            for (u, v) in matching:
                if u > v:
                    u, v = v, u
                ei = edge_idx[(u, v)]
                best_colors[ei] = match_idx
    
    # Flatten to list in lexicographic order
    result = best_colors[:]
    return result
