"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # We want to minimize the number of distinct colors used.
    # The seed program uses a greedy approach that picks the smallest valid color index.
    # This tends to use few colors but might get stuck or use more than necessary if a different assignment was possible.
    # However, the problem asks to MINIMIZE the number of labels.
    # The seed program's logic:
    # 1. For each position x (1..128), try colors 0..127.
    # 2. Condition 1: For every 4-term AP ending at x (i.e., x-3d, x-2d, x-d, x), the set of colors must have size >= 3.
    #    This means among {c_x, c_{x-d}, c_{x-2d}, c_{x-3d}}, at least 3 must be distinct.
    #    This is equivalent to NOT (all 4 same) AND NOT (exactly 2 distinct colors in a 2-2 split? No, just size >= 3).
    #    Wait, "at least 3 distinct colors" means the number of unique values in the 4 positions is >= 3.
    #    So it forbids:
    #    - 4 same (1 distinct)
    #    - 3 same, 1 diff (2 distinct)
    #    - 2 same, 2 same (2 distinct)
    #    - 2 same, 1 diff, 1 diff (3 distinct) -> OK
    #    So it only allows 3 or 4 distinct colors.
    
    # Condition 2 in seed: `any(colors[x-1-j*d]!=color for j in (1,2))`
    # This checks positions x-1-d and x-1-2d? No, j is 1,2.
    # Positions: x-1-1*d = x-1-d, x-1-2*d = x-1-2d.
    # This condition seems to be a specific constraint on a 3-term AP? Or part of a 4-term?
    # Let's re-read the task: "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
    # The seed's condition 1 checks 4-term APs: x-3d, x-2d, x-d, x.
    # The seed's condition 2 checks... `any(colors[x-1-j*d]!=color for j in (1,2))`.
    # If d=1, positions are x-2, x-3. It checks if c[x-1] != c[x-2] OR c[x-1] != c[x-3].
    # This implies that c[x-1] cannot be equal to BOTH c[x-2] and c[x-3].
    # This prevents the pattern A A B in positions x-3, x-2, x-1?
    # If c[x-3]=A, c[x-2]=A, c[x-1]=B. Then c[x-1]!=c[x-2] is False, c[x-1]!=c[x-3] is False. So any() is False.
    # So it forbids c[x-3]==c[x-2]==c[x-1]? No, it forbids c[x-1]==c[x-2] AND c[x-1]==c[x-3].
    # So it forbids c[x-1] being equal to both previous ones.
    # This is a constraint on 3-term APs? Or just a heuristic to prevent bad patterns?
    # Actually, if we have a 4-term AP x-3d, x-2d, x-d, x.
    # If we pick c[x] such that it creates only 2 distinct colors in the AP, it's invalid.
    # The seed's condition 1 explicitly checks this for the full 4-term AP.
    # Why condition 2?
    # Maybe it's checking 3-term APs ending at x-1?
    # If the task only requires 4-term APs to have >=3 colors, then 3-term APs can have 1 or 2 colors.
    # However, if a 3-term AP has 2 colors (e.g., A A B), and we extend it to a 4-term AP (A A B C),
    # we need {A, B, C} to have >= 3 distinct colors. So C must be different from A and B?
    # If C=A, we have {A, B} (2 distinct). Invalid.
    # If C=B, we have {A, B} (2 distinct). Invalid.
    # So if we have A A B, the next term MUST be a third color.
    # The seed's condition 2 seems to enforce something related to this.
    
    # Let's optimize the seed's greedy approach.
    # The seed is O(N * 128 * N) roughly. N=128. 128*128*128 = 2M ops. Very fast.
    # The bottleneck is likely not speed, but the quality of the greedy choice.
    # The greedy choice picks the SMALLEST valid color index.
    # This minimizes the max color index used, but does it minimize the COUNT of distinct colors?
    # Not necessarily. Using small indices might force us to use many distinct small indices,
    # whereas using a few large indices might allow reusing them more effectively?
    # No, the goal is to minimize the NUMBER of distinct labels.
    # If I use labels {0, 1, 2}, that's 3 labels.
    # If I use labels {0, 10, 20}, that's 3 labels.
    # The evaluator measures "raw value (lower is better)". The task says "Minimize the number of labels actually used".
    # So the score is the count of distinct colors in the output array.
    
    # The seed program likely produces a specific count.
    # Can we do better?
    # If the minimum possible is K, and the seed finds K, we can't beat it.
    # If the seed finds K+1, we can try to find K.
    
    # Strategy:
    # 1. Run the seed's greedy algorithm to get a baseline coloring and count.
    # 2. Try to improve it using local search (swap colors, recolor) to reduce the distinct count.
    #    - Pick a color that is used least frequently.
    #    - Try to map all its occurrences to other existing colors.
    #    - If successful, remove that color.
    #    - Repeat until no more colors can be removed.
    
    def get_distinct_count(colors):
        return len(set(colors))
    
    def is_valid_4ap(colors, i, d):
        # Check 4-term AP ending at i with step d: i-3d, i-2d, i-d, i
        # Indices must be >= 0
        if i - 3*d < 0:
            return True
        c0 = colors[i-3*d]
        c1 = colors[i-2*d]
        c2 = colors[i-d]
        c3 = colors[i]
        return len({c0, c1, c2, c3}) >= 3
    
    def is_valid_coloring(colors):
        n = len(colors)
        for i in range(3, n):
            max_d = (i) // 3
            for d in range(1, max_d + 1):
                if not is_valid_4ap(colors, i, d):
                    return False
        return True
    
    # Seed Greedy with Bitmask Optimization and 3rd-term constraint
    def greedy_construct():
        colors = [0] * 128
        for i in range(128):
            forbidden_mask = 0
            stuck = False
            
            # Check constraints where i is the 4th term (i-3d, i-2d, i-d, i)
            max_d4 = i // 3
            for d in range(1, max_d4 + 1):
                c0 = colors[i-3*d]
                c1 = colors[i-2*d]
                c2 = colors[i-d]
                s = {c0, c1, c2}
                if len(s) == 1:
                    # This should not happen if we prevent monochromatic 3-term APs
                    stuck = True
                    break
                elif len(s) == 2:
                    for c in s:
                        forbidden_mask |= (1 << c)
                # if len(s) == 3, no restriction on c from this AP
            
            if stuck:
                # Fallback: should not happen. Pick 0.
                colors[i] = 0
                continue

            # Check constraints where i is the 3rd term (i-2d, i-d, i, i+d)
            # i is 3rd term => i = a+2d => a = i-2d.
            # We need i+d < 128.
            max_d3 = min(i // 2, (127 - i))
            for d in range(1, max_d3 + 1):
                c_prev1 = colors[i-2*d]
                c_prev2 = colors[i-d]
                if c_prev1 == c_prev2:
                    forbidden_mask |= (1 << c_prev1)

            # Find smallest available color
            for c in range(128):
                if not (forbidden_mask & (1 << c)):
                    colors[i] = c
                    break
            else:
                # No color found? Should not happen with 128 colors.
                colors[i] = 0
        
        return colors
    
    c1 = greedy_construct()
    count1 = get_distinct_count(c1)
    
    # Local Search to reduce color count
    # Try to merge colors.
    # Start with the current set of colors.
    # While possible:
    #   Pick a color C to remove (e.g., the one with fewest occurrences).
    #   Try to recolor all positions with C to other colors such that validity is maintained.
    #   This is a CSP. We can use backtracking or simple greedy recoloring.
    
    def try_remove_color(colors, target_color):
        # Returns new colors if target_color can be removed, else None
        n = len(colors)
        positions = [i for i, c in enumerate(colors) if c == target_color]
        if not positions:
            return None
            
        # Create a copy
        new_colors = colors[:]
        
        # We need to assign new colors to these positions.
        # Available colors: all colors in `colors` except target_color.
        available = [c for c in set(colors) if c != target_color]
        
        # If no available colors, we can't remove it (unless we introduce new ones, but that doesn't reduce count)
        if not available:
            return None
            
        # Backtracking search for the positions of target_color
        # Order positions by most constrained? Or just in order.
        
        def backtrack(idx, current_colors):
            if idx == len(positions):
                # Check full validity
                if is_valid_coloring(current_colors):
                    return True
                return False
                
            pos = positions[idx]
            for new_c in available:
                if new_c == current_colors[pos]:
                    continue
                # Tentatively assign
                old_c = current_colors[pos]
                current_colors[pos] = new_c
                
                # Prune: check local constraints involving pos
                # 4-term APs ending at pos
                valid_local = True
                max_d = pos // 3
                for d in range(1, max_d + 1):
                    c0 = current_colors[pos-3*d]
                    c1 = current_colors[pos-2*d]
                    c2 = current_colors[pos-d]
                    c3 = new_c
                    if len({c0, c1, c2, c3}) < 3:
                        valid_local = False
                        break
                # 4-term APs starting at pos? No, 4-term AP is defined by 4 terms.
                # If pos is the first term: pos, pos+d, pos+2d, pos+3d.
                # If pos is the second term: pos-d, pos, pos+d, pos+2d.
                # If pos is the third term: pos-2d, pos-d, pos, pos+d.
                # If pos is the fourth term: pos-3d, pos-2d, pos-d, pos.
                # We checked "ending at pos" (pos is 4th term).
                # We also need to check if pos is 1st, 2nd, or 3rd term.
                
                # Pos as 1st term: pos, pos+d, pos+2d, pos+3d
                for d in range(1, (n-1-pos)//3 + 1):
                    c0 = new_c
                    c1 = current_colors[pos+d]
                    c2 = current_colors[pos+2*d]
                    c3 = current_colors[pos+3*d]
                    if len({c0, c1, c2, c3}) < 3:
                        valid_local = False
                        break
                if not valid_local:
                    current_colors[pos] = old_c
                    continue
                    
                # Pos as 2nd term: pos-d, pos, pos+d, pos+2d
                for d in range(1, min(pos, (n-1-pos)//2) + 1):
                    c0 = current_colors[pos-d]
                    c1 = new_c
                    c2 = current_colors[pos+d]
                    c3 = current_colors[pos+2*d]
                    if len({c0, c1, c2, c3}) < 3:
                        valid_local = False
                        break
                if not valid_local:
                    current_colors[pos] = old_c
                    continue
                    
                # Pos as 3rd term: pos-2d, pos-d, pos, pos+d
                for d in range(1, min(pos//2, (n-1-pos)) + 1):
                    c0 = current_colors[pos-2*d]
                    c1 = current_colors[pos-d]
                    c2 = new_c
                    c3 = current_colors[pos+d]
                    if len({c0, c1, c2, c3}) < 3:
                        valid_local = False
                        break
                if not valid_local:
                    current_colors[pos] = old_c
                    continue
                    
                if backtrack(idx + 1, current_colors):
                    return True
                    
                current_colors[pos] = old_c
                
            return False
            
        if backtrack(0, new_colors):
            return new_colors
        return None

    current_colors = c1
    current_count = count1
    
    # Iterate removal
    improved = True
    while improved:
        improved = False
        colors_in_use = list(set(current_colors))
        # Sort by frequency (ascending) to try removing rare colors first
        colors_in_use.sort(key=lambda c: current_colors.count(c))
        
        for c in colors_in_use:
            if time.time() > deadline:
                break
            new_colors = try_remove_color(current_colors, c)
            if new_colors is not None:
                current_colors = new_colors
                current_count = get_distinct_count(current_colors)
                improved = True
                break # Restart loop as colors changed
                
        if time.time() > deadline:
            break

    return current_colors
# EVOLVE-BLOCK-END
