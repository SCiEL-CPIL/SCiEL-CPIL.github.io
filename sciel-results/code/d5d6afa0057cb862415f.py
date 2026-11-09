"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Precompute all 4-term progressions
    progressions = []
    for a in range(1, n+1):
        for d in range(1, n):
            if a + 3*d <= n:
                progressions.append((a-1, a-1+d, a-1+2*d, a-1+3*d))  # 0-indexed
    
    num_prog = len(progressions)
    
    def count_colors(colors):
        return len(set(colors))
    
    def check_violations(colors):
        """Return list of violated progression indices."""
        violations = []
        for i, (a, b, c, d) in enumerate(progressions):
            colors_set = {colors[a], colors[b], colors[c], colors[d]}
            if len(colors_set) < 3:
                violations.append(i)
        return violations
    
    def is_valid(colors):
        for (a, b, c, d) in progressions:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    # Try different numbers of colors, starting from 3
    best_colors = None
    best_count = n  # worst case
    
    for num_c in range(3, 10):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            colors = np.random.randint(0, num_c, size=n)
            
            # Local search: fix violations
            max_iter = 5000
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                violations = check_violations(colors.tolist())
                if not violations:
                    # Valid!
                    actual_colors = len(set(colors.tolist()))
                    if actual_colors < best_count:
                        best_count = actual_colors
                        best_colors = colors.tolist()
                    found = True
                    break
                
                # Pick a random violated progression and fix it
                vi = random.choice(violations)
                a, b, c, d = progressions[vi]
                
                # Try changing one of the four positions
                positions = [a, b, c, d]
                random.shuffle(positions)
                
                fixed = False
                for pos in positions:
                    # Try changing this position to a different color
                    current = colors[pos]
                    for new_color in range(num_c):
                        if new_color == current:
                            continue
                        colors[pos] = new_color
                        # Quick check: does this fix the violation without creating new ones nearby?
                        if len({colors[a], colors[b], colors[c], colors[d]}) >= 3:
                            fixed = True
                            break
                    if fixed:
                        break
                    colors[pos] = current  # revert
                
                if not fixed:
                    # Try a random change
                    pos = random.choice(positions)
                    new_color = random.randint(0, num_c - 1)
                    colors[pos] = new_color
            
            if found:
                break
        
        if found and best_count <= num_c:
            break
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        # Remap colors to be in 0..127 range (they already are)
        return best_colors
    
    # Fallback: use the greedy approach from seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                if x-1-3*d >= 0:
                    prog_colors = {color, colors[x-1-d], colors[x-1-2*d], colors[x-1-3*d]}
                    if len(prog_colors) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)  # fallback
    
    return colors
# EVOLVE-BLOCK-END
