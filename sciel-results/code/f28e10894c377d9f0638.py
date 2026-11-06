"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    # AP: (i, i+d, i+2d, i+3d) where d >= 1, all indices < n
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v
    
    def is_ap_valid(colors, ap):
        c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
        return len(c) >= 3
    
    def greedy_construct():
        """Deterministic greedy: assign smallest valid color at each position."""
        colors = [None] * n
        for x in range(n):
            for color in range(128):
                valid = True
                # Check all APs that include position x
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    # Build the set of colors in this AP, replacing x's color with candidate
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
            if colors[x] is None:
                # Fallback: should not happen with 128 colors
                colors[x] = x
        return colors
    
    def try_reduce_colors_aggressive(colors, time_limit):
        """
        Aggressively try to reduce colors using constraint propagation.
        For each color to remove, check if it can be remapped by solving
        a constraint satisfaction problem.
        """
        colors = colors[:]
        used_colors = sorted(set(colors))
        current_count = len(used_colors)
        
        if current_count <= 3:
            return colors, current_count
        
        best_colors = colors[:]
        best_count = current_count
        
        # Sort colors by frequency (least frequent first, easier to remove)
        color_freq = {}
        for c in colors:
            color_freq[c] = color_freq.get(c, 0) + 1
        sorted_colors = sorted(used_colors, key=lambda c: color_freq[c])
        
        for color_to_remove in sorted_colors:
            if time.time() > time_limit:
                break
            if best_count <= 3:
                break
            
            positions = [i for i in range(n) if colors[i] == color_to_remove]
            if not positions:
                continue
            
            # Try remapping all positions of color_to_remove to a single target color
            for target_color in sorted_colors:
                if target_color == color_to_remove:
                    continue
                
                # Temporarily remap
                old_vals = {}
                for pos in positions:
                    old_vals[pos] = colors[pos]
                    colors[pos] = target_color
                
                v = count_violations(colors)
                if v == 0:
                    best_colors = colors[:]
                    best_count -= 1
                    break
                else:
                    # Revert
                    for pos in positions:
                        colors[pos] = old_vals[pos]
            
            # Local search: assign each position in 'positions' a color from remaining colors
            if time.time() > time_limit:
                break
            
            remaining_colors = [c for c in sorted_colors if c != color_to_remove]
            if not remaining_colors:
                continue
            
            # Initialize with random assignment
            assign = [random.choice(remaining_colors) for _ in positions]
            current_v = 999999
            best_assign = assign[:]
            best_v = 999999
            
            for iteration in range(2000):
                if time.time() > time_limit:
                    break
                
                idx = random.randint(0, len(positions) - 1)
                pos = positions[idx]
                old_c = assign[idx]
                new_c = random.choice(remaining_colors)
                if new_c == old_c:
                    continue
                
                assign[idx] = new_c
                
                for i, p in enumerate(positions):
                    colors[p] = assign[i]
                v = count_violations(colors)
                
                if v < best_v:
                    best_v = v
                    best_assign = assign[:]
                    if v == 0:
                        break
                
                if v > current_v:
                    assign[idx] = old_c
                else:
                    current_v = v
                
                for i, p in enumerate(positions):
                    colors[p] = old_vals[p] if p in old_vals else colors[p]
            
            if best_v == 0:
                for i, p in enumerate(positions):
                    colors[p] = best_assign[i]
                best_colors = colors[:]
                best_count -= 1
            else:
                for i, p in enumerate(positions):
                    colors[p] = old_vals[p]
        
        return best_colors, best_count
    
    def local_search_refine(colors, time_limit):
        """Simulated annealing to fix violations and minimize color count."""
        colors = colors[:]
        current_violations = count_violations(colors)
        current_colors_used = len(set(colors))
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1
        
        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        temp = 3.0
        cooling = 0.997
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Determine candidate colors: current used colors + a few new ones
            used = set(colors)
            candidates = list(used)
            # Add some random new colors to explore
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)
            
            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0
            
            affected_aps = pos_to_aps[pos]
            
            # Count current violations among affected APs
            curr_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    curr_invalid += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid += 1
                
                delta_v = new_invalid - curr_invalid
                # Efficient color count change: if old_color count was 1, removing it reduces count
                # if new_color count was 0, adding it increases count
                delta_c = 0
                if color_counts[old_color] == 1:
                    delta_c -= 1
                if color_counts[new_color] == 0:
                    delta_c += 1
                
                # Score: prioritize fixing violations, then reducing colors
                score = delta_v * 100 + delta_c
                best_score = best_delta_v * 100 + best_delta_c
                
                if score < best_score:
                    best_score = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
            
            colors[pos] = old_color
            
            # Accept based on simulated annealing
            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                # Compute acceptance probability
                energy = best_delta_v * 100 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
            
            if accept:
                colors[pos] = best_new_color
                color_counts[old_color] -= 1
                color_counts[best_new_color] += 1
                current_violations += best_delta_v
                current_colors_used = sum(1 for c in range(128) if color_counts[c] > 0)
                
                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]
                
                if current_violations == 0 and current_colors_used <= 3:
                    break
            
            temp *= cooling
        
        return best_colors, best_violations, best_colors_used
    
    # Step 1: Greedy construction
    colors = greedy_construct()
    initial_violations = count_violations(colors)
    initial_colors = len(set(colors))
    
    # Step 2: Aggressive color reduction
    if initial_violations == 0:
        colors, reduced_count = try_reduce_colors_aggressive(colors, time.time() + 20.0)
    
    # Step 3: Local search refinement if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        colors, final_violations, final_colors = local_search_refine(colors, deadline - 1.0)
    else:
        final_violations = initial_violations
        final_colors = initial_colors
    
    # Ensure we return a valid solution
    if final_violations > 0:
        # Fallback: identity mapping is always valid
        return list(range(n))
    
    return colors
# EVOLVE-BLOCK-END
