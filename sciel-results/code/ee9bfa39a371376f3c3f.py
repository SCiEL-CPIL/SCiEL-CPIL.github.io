"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    # Precompute all 4-term APs: (a, b, c, d) where b=a+d, c=a+2d, d=a+3d, d>0
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    def count_violations(colors):
        """Count 4-APs using fewer than 3 distinct colors."""
        violations = 0
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                violations += 1
        return violations
    
    def count_violations_incremental(colors, pos, old_color, new_color):
        """Count violations after changing position pos from old_color to new_color."""
        if old_color == new_color:
            return count_violations(colors)
        colors[pos] = new_color
        v = count_violations(colors)
        colors[pos] = old_color
        return v
    
    best_colors = None
    best_violations = float('inf')
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with random 3-coloring
        colors = np.random.randint(0, 3, size=n)
        v = count_violations(colors)
        
        if v < best_violations:
            best_violations = v
            best_colors = colors.copy()
        
        if v == 0:
            break
        
        # Local search: try flipping positions
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try each position, each color
            for pos in range(n):
                if time.time() > deadline:
                    break
                old = colors[pos]
                for new_c in range(3):
                    if new_c == old:
                        continue
                    colors[pos] = new_c
                    new_v = count_violations(colors)
                    if new_v < v:
                        v = new_v
                        improved = True
                        if v == 0:
                            break
                    else:
                        colors[pos] = old
                if v == 0:
                    break
            
            if v < best_violations:
                best_violations = v
                best_colors = colors.copy()
    
    if best_violations == 0 and best_colors is not None:
        # Map to consecutive labels 0,1,2
        used = sorted(set(best_colors.tolist()))
        mapping = {old: new for new, old in enumerate(used)}
        return [mapping[c] for c in best_colors.tolist()]
    
    # If no perfect solution found, return best effort
    if best_colors is not None:
        used = sorted(set(best_colors.tolist()))
        mapping = {old: new for new, old in enumerate(used)}
        return [mapping[c] for c in best_colors.tolist()]
    
    # Fallback: greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
# EVOLVE-BLOCK-END
