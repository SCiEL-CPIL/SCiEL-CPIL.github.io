import numpy as np
import random

def construct():
    N = 128
    
    # Precompute all nonconstant 4-term arithmetic progressions
    # positions are 1-indexed in the problem, but we use 0-indexed internally
    # A 4-term AP is (a, a+d, a+2d, a+3d) with d >= 1
    # We need to ensure that no such AP has at most 2 distinct colors
    
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # We'll use a greedy + local search approach
    # Start with a random assignment and try to minimize the number of colors used
    # while satisfying the constraint that every AP uses at least 3 distinct colors
    
    best_colors = None
    best_num_colors = N + 1
    
    # Try multiple random restarts
    for restart in range(200):
        random.seed(restart)
        
        # Start with a simple pattern: alternate between 2 colors won't work
        # Let's start with a random assignment of 3 colors
        # and then try to reduce colors while maintaining validity
        
        # Strategy: start with all positions colored with 3 colors in a pattern,
        # then try to merge colors where possible
        
        # Initial coloring: use 3 colors in a cycling pattern
        colors = np.zeros(N, dtype=int)
        for i in range(N):
            colors[i] = i % 3
        
        # Check if this is valid
        def is_valid(colors):
            for a, b, c, d in aps:
                vals = {colors[a], colors[b], colors[c], colors[d]}
                if len(vals) < 3:
                    return False
            return True
        
        if not is_valid(colors):
            continue
        
        # Try to reduce the number of colors by merging
        # We'll try to change individual positions to reduce the color count
        num_iterations = 0
        max_iterations = 10000
        
        while num_iterations < max_iterations:
            num_iterations += 1
            improved = False
            
            # Get current color labels
            current_labels = set(colors)
            min_label = min(current_labels)
            max_label = max(current_labels)
            
            # Try to reassign each position to a lower label if possible
            for i in range(N):
                old_color = colors[i]
                # Try to change to each color from 0 to max_label
                for new_color in range(0, max_label + 1):
                    if new_color == old_color:
                        continue
                    colors[i] = new_color
                    if is_valid(colors):
                        improved = True
                        break
                    colors[i] = old_color
            
            if not improved:
                break
        
        # Count distinct colors used
        distinct = len(set(colors))
        
        if distinct < best_num_colors:
            best_num_colors = distinct
            best_colors = colors.copy()
            if best_num_colors <= 3:
                break
    
    # If we found a valid coloring, return it; otherwise fallback
    if best_colors is not None:
        # Remap colors to 0..k-1 range
        unique_colors = sorted(set(best_colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        result = [color_map[c] for c in best_colors]
        return result
    else:
        # Fallback: use the simple 3-color pattern
        result = [i % 3 for i in range(N)]
        return result
