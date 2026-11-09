import time
import numpy as np
from itertools import product

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

def _validate(coloring):
    """Check if a coloring satisfies the constraint for all 4-term APs."""
    for ap in _ALL_APS:
        if len({coloring[i] for i in ap}) < 3:
            return False
    return True

def _find_periodic_pattern(period, k_colors, deadline):
    """
    Exhaustively search for a periodic pattern of given period and k_colors
    that satisfies the 3-color progression constraint for all x <= 128.
    Returns a list of 128 colors if found, else None.
    """
    # Enumerate all possible patterns of length 'period' with k_colors
    # Use symmetry breaking: fix pattern[0]=0, and require pattern to use all k_colors
    
    # For small periods, enumerate all assignments
    if period <= 10:
        # Generate all patterns with pattern[0]=0
        # Use colors 0..k_colors-1
        for rest in product(range(k_colors), repeat=period-1):
            pattern = [0] + list(rest)
            
            # Check if pattern uses at least 3 colors (required for any AP to have 3 colors)
            if len(set(pattern)) < 3:
                continue
            
            # Check if this periodic pattern satisfies all APs for positions 0..127
            valid = True
            for ap in _ALL_APS:
                colors_in_ap = {pattern[i % period] for i in ap}
                if len(colors_in_ap) < 3:
                    valid = False
                    break
            
            if valid:
                # Expand to 128 positions
                expanded = [pattern[i % period] for i in range(128)]
                return expanded
            
            # Time check
            if time.time() > deadline:
                return None
    
    return None

def _local_search_reduce_colors(coloring, deadline):
    """
    Try to reduce the number of distinct colors in a valid coloring
    by reassigning positions to existing colors.
    """
    coloring = list(coloring)
    n = len(coloring)
    
    def get_used_colors(arr):
        return sorted(set(arr))
    
    used = get_used_colors(coloring)
    current_count = len(used)
    
    best = coloring[:]
    best_count = current_count
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        used = get_used_colors(best)
        
        for color_to_remove in used:
            if time.time() > deadline:
                break
            
            if len(used) <= 3:
                break
            
            trial = best[:]
            positions = [i for i in range(n) if trial[i] == color_to_remove]
            
            if not positions:
                continue
            
            success = True
            for pos in positions:
                original = trial[pos]
                found_new = False
                for new_color in used:
                    if new_color == color_to_remove:
                        continue
                    trial[pos] = new_color
                    if _validate(trial):
                        found_new = True
                        break
                    trial[pos] = original
                
                if not found_new:
                    success = False
                    break
            
            if success:
                new_used = get_used_colors(trial)
                if len(new_used) < len(get_used_colors(best)):
                    best = trial
                    best_count = len(new_used)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best, best_count

def construct():
    start = time.time()
    DEADLINE = 80.0  # safety margin before 90s limit
    
    n = 128
    
    # Precompute position -> APs mapping for faster local checks
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)
    
    best = None
    best_ncolors = n
    
    # Try to find a periodic pattern with 3 colors
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]:
        if time.time() > DEADLINE - 15:
            break
        
        pattern = _find_periodic_pattern(period, 3, DEADLINE - 15)
        if pattern is not None:
            best = pattern
            best_ncolors = 3
            break
    
    # If no periodic pattern found, use a randomized local search approach
    if best is None:
        # Start with a random 3-color assignment
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > DEADLINE - 10:
                break
            
            colors = [random.randint(0, 2) for _ in range(n)]
            
            # Local search: try to fix violations by changing individual positions
            improved = True
            while improved and time.time() < DEADLINE - 10:
                improved = False
                
                # Find all violating APs
                violations = []
                for ap in _ALL_APS:
                    if len({colors[i] for i in ap}) < 3:
                        violations.append(ap)
                
                if not violations:
                    best = colors
                    best_ncolors = 3
                    break
                
                # Pick a random violation and try to fix it
                ap = random.choice(violations)
                current_colors = {colors[i] for i in ap}
                
                # Try changing one position in this AP to a new color
                pos_to_change = random.choice(ap)
                old_color = colors[pos_to_change]
                
                for new_color in range(3):
                    if new_color == old_color:
                        continue
                    colors[pos_to_change] = new_color
                    
                    # Check if this creates new violations
                    creates_violation = False
                    for other_ap in pos_in_aps[pos_to_change]:
                        if len({colors[i] for i in other_ap}) < 3:
                            creates_violation = True
                            break
                    
                    if not creates_violation:
                        improved = True
                        break
                    else:
                        colors[pos_to_change] = old_color
            
            if best is not None:
                break
    
    # Fallback: try with 4 colors if 3 colors failed
    if best is None:
        import random
        random.seed(43)
        
        for attempt in range(20):
            if time.time() > DEADLINE - 5:
                break
            
            colors = [random.randint(0, 3) for _ in range(n)]
            
            improved = True
            while improved and time.time() < DEADLINE - 5:
                improved = False
                
                violations = []
                for ap in _ALL_APS:
                    if len({colors[i] for i in ap}) < 3:
                        violations.append(ap)
                
                if not violations:
                    best = colors
                    best_ncolors = len(set(colors))
                    break
                
                ap = random.choice(violations)
                pos_to_change = random.choice(ap)
                old_color = colors[pos_to_change]
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    colors[pos_to_change] = new_color
                    
                    creates_violation = False
                    for other_ap in pos_in_aps[pos_to_change]:
                        if len({colors[i] for i in other_ap}) < 3:
                            creates_violation = True
                            break
                    
                    if not creates_violation:
                        improved = True
                        break
                    else:
                        colors[pos_to_change] = old_color
    
    # If still no solution, use greedy construction
    if best is None:
        colors = []
        for x in range(n):
            placed = False
            for color in range(128):
                valid = True
                for ap, others in [(ap, [i for i in ap if i != x]) for ap in pos_in_aps[x]]:
                    cs = {color}
                    for o in others:
                        if o < len(colors):
                            cs.add(colors[o])
                        else:
                            break
                    if len(cs) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    placed = True
                    break
            if not placed:
                colors.append(0)
        best = colors
        best_ncolors = len(set(colors))
    
    # Try to reduce colors further via local search
    if best is not None and best_ncolors > 3 and time.time() < DEADLINE - 5:
        improved, improved_count = _local_search_reduce_colors(best, DEADLINE - 5)
        if improved_count < best_ncolors:
            best = improved
            best_ncolors = improved_count
    
    # Ensure valid return
    if best is not None:
        return best
    else:
        # Absolute fallback
        return list(range(n))
