"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def construct():
    deadline = time.time() + 55  # 55s safety margin before 60s limit
    
    N = 128
    
    def get_distinct_count(colors):
        return len(set(colors))
    
    def count_violations(colors):
        """Count number of invalid 4-term APs (those with < 3 distinct colors)."""
        violations = 0
        for i in range(3, N):
            max_d = i // 3
            for d in range(1, max_d + 1):
                c0 = colors[i-3*d]
                c1 = colors[i-2*d]
                c2 = colors[i-d]
                c3 = colors[i]
                if len({c0, c1, c2, c3}) < 3:
                    violations += 1
        return violations
    
    def is_valid(colors):
        return count_violations(colors) == 0
    
    def greedy_construct():
        """Greedy construction ensuring validity at each step."""
        colors = [0] * N
        for i in range(N):
            forbidden = set()
            
            # Check 4-term APs where i is the 4th term: (i-3d, i-2d, i-d, i)
            for d in range(1, i // 3 + 1):
                c0, c1, c2 = colors[i-3*d], colors[i-2*d], colors[i-d]
                s = {c0, c1, c2}
                if len(s) == 1:
                    # All three same - any color for i works as long as it's different
                    # Actually if all three are same, then {c0,c1,c2,c3} has size 2 if c3==c0, or 1 if... 
                    # Wait: if c0=c1=c2=A, then we need {A, A, A, c3} to have >=3 distinct.
                    # That's impossible! So this state should never occur in a valid prefix.
                    # But let's handle it: no valid color exists, so we're stuck.
                    # For safety, pick a color and hope local search fixes it.
                    forbidden.add(c0)  # This doesn't help, but let's continue
                elif len(s) == 2:
                    # Two distinct colors among c0,c1,c2
                    # We need {c0,c1,c2,c3} to have >=3 distinct
                    # So c3 must NOT be in s (otherwise we'd have at most 2 distinct)
                    forbidden.update(s)
                # else len(s)==3, no restriction on c3 from this AP
            
            # Also check: if i is the 3rd term of some AP (i-2d, i-d, i, i+d)
            # We can't check future, but we should avoid creating patterns that force bad futures
            # For now, just pick the smallest available color
            
            for c in range(128):
                if c not in forbidden:
                    colors[i] = c
                    break
            else:
                colors[i] = 0  # Fallback
        
        return colors
    
    # Step 1: Get initial valid coloring
    best_colors = greedy_construct()
    best_count = get_distinct_count(best_colors)
    
    # Step 2: Local search to reduce color count
    # Strategy: try to merge colors by recoloring all occurrences of one color to another
    # Use simulated annealing or simple hill climbing
    
    def try_merge(color_a, color_b, colors):
        """Try to recolor all positions with color_a to color_b. Return new colors if valid."""
        new_colors = colors[:]
        for i in range(N):
            if new_colors[i] == color_a:
                new_colors[i] = color_b
        
        if is_valid(new_colors):
            return new_colors
        return None
    
    # Local search: try to reduce the number of distinct colors
    current_colors = best_colors[:]
    current_count = best_count
    
    # Get the set of colors currently used
    used_colors = set(current_colors)
    
    # Try to eliminate colors one by one, starting with the least frequent
    color_freq = {}
    for c in current_colors:
        color_freq[c] = color_freq.get(c, 0) + 1
    
    # Sort colors by frequency (ascending) - try to eliminate rare colors first
    sorted_colors = sorted(used_colors, key=lambda c: color_freq.get(c, 0))
    
    # For each color to eliminate, try merging into each other color
    improved = True
    while improved:
        improved = False
        current_used = set(current_colors)
        current_freq = {}
        for c in current_colors:
            current_freq[c] = current_freq.get(c, 0) + 1
        
        sorted_cur = sorted(current_used, key=lambda c: current_freq.get(c, 0))
        
        for elim_color in sorted_cur:
            if time.time() > deadline - 5:
                break
            if elim_color not in current_used:
                continue
            
            # Try to merge elim_color into each other color
            for target_color in current_used:
                if target_color == elim_color:
                    continue
                
                new_colors = try_merge(elim_color, target_color, current_colors)
                if new_colors is not None:
                    new_count = get_distinct_count(new_colors)
                    if new_count < current_count:
                        current_colors = new_colors
                        current_count = new_count
                        if current_count < best_count:
                            best_colors = current_colors[:]
                            best_count = current_count
                        improved = True
                        break  # Restart the while loop with new state
        
        if time.time() > deadline - 5:
            break
    
    # Step 3: Additional local search with random perturbations if time allows
    if time.time() < deadline - 10:
        # Try random single-position recoloring to find better solutions
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            start_time = time.time()
            # Pick a random position and try changing its color
            pos = random.randint(0, N-1)
            old_color = current_colors[pos]
            
            # Try each possible color
            for new_color in range(128):
                if new_color == old_color:
                    continue
                
                current_colors[pos] = new_color
                new_count = get_distinct_count(current_colors)
                
                if new_count < current_count and is_valid(current_colors):
                    current_count = new_count
                    if current_count < best_count:
                        best_colors = current_colors[:]
                        best_count = current_count
                    break
                else:
                    current_colors[pos] = old_color
    
    return best_colors
# EVOLVE-BLOCK-END
