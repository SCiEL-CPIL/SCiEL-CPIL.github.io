"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (i, i+d, i+2d, i+3d) within 0..n-1
    progressions = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            progressions.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_progs = len(progressions)
    
    # For each position, store indices of progressions it's in and the other 3 positions
    pos_progs = [[] for _ in range(n)]
    for pi, (a, b, c, e) in enumerate(progressions):
        for pos in (a, b, c, e):
            others = [p for p in (a, b, c, e) if p != pos]
            pos_progs[pos].append((pi, others))
    
    def count_colors(arr):
        return len(set(arr))
    
    def validate_full(arr):
        """Full validation - check all progressions."""
        for (a, b, c, e) in progressions:
            if len({arr[a], arr[b], arr[c], arr[e]}) < 3:
                return False
        return True
    
    def validate_change(arr, pos, old_color, new_color):
        """Check if changing arr[pos] from old_color to new_color maintains validity.
        Only checks progressions containing pos."""
        for (pi, others) in pos_progs[pos]:
            # Get colors of the other 3 positions in this progression
            prog_colors = set()
            prog_colors.add(new_color)
            for o in others:
                prog_colors.add(arr[o])
            if len(prog_colors) < 3:
                return False
        return True
    
    def greedy_construct():
        """Fast greedy construction of a valid coloring."""
        colors = []
        for x in range(1, n + 1):
            pos = x - 1  # 0-indexed
            found = False
            for color in range(n):
                valid = True
                # Check all progressions where pos is the last term (i+3d = pos)
                # pos = i + 3d => i = pos - 3d
                max_d_last = pos // 3
                for d in range(1, max_d_last + 1):
                    i = pos - 3 * d
                    if i < 0:
                        continue
                    c0 = colors[i]
                    c1 = colors[i + d]
                    c2 = colors[i + 2 * d]
                    c3 = color
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check progressions where pos is the 3rd term (i+2d = pos)
                # pos = i + 2d => i = pos - 2d
                max_d_3rd = pos // 2
                for d in range(1, max_d_3rd + 1):
                    i = pos - 2 * d
                    if i < 0:
                        continue
                    # Terms: i, i+d, i+2d=pos, i+3d
                    # i+3d = pos + d, which may be >= n
                    c0 = colors[i]
                    c1 = colors[i + d]
                    c2 = color
                    # The 4th term i+3d = pos+d may not be assigned yet
                    # We need to ensure that even in the worst case, we can have >=3 colors
                    # If c0 == c1 == c2, then no matter what the 4th term is, we have at most 2 colors
                    if c0 == c1 and c1 == c2:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check progressions where pos is the 2nd term (i+d = pos)
                # pos = i + d => i = pos - d
                # Terms: i, i+d=pos, i+2d, i+3d
                # i+2d = pos + d, i+3d = pos + 2d (future terms)
                # We need to ensure the first two terms aren't both same as what would force issues
                # Actually, if pos is 2nd term, the constraint is just that all 4 have >=3 colors.
                # Since future terms aren't assigned, we just need to not create an impossible situation.
                # The only impossible situation is if the already-assigned terms force < 3 colors.
                # With 2 assigned terms (i and pos), we need at least one of the future terms to differ.
                # This is always possible unless both future terms are forced to be one of {c_i, c_pos}.
                # For safety, we don't add extra constraints here - the greedy will handle it.
                
                colors.append(color)
                found = True
                break
            if not found:
                # Should not happen with n colors available
                colors.append(0)
        return colors
    
    # Start with greedy construction
    colors = greedy_construct()
    
    # Verify it's valid
    if not validate_full(colors):
        # Fallback: use even more colors
        colors = list(range(n))
    
    best_colors = colors[:]
    best_count = count_colors(colors)
    
    # Local search to reduce color count
    def local_search(arr, max_iterations=200000):
        """Improve coloring by changing individual positions to reduce color count."""
        arr = list(arr)
        n_local = len(arr)
        current_colors = count_colors(arr)
        
        # Build color frequency
        color_freq = {}
        for c in arr:
            color_freq[c] = color_freq.get(c, 0) + 1
        
        used_colors = list(color_freq.keys())
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try to reduce color count by reassigning positions
            # Strategy: for each color, try to reassign all its positions to other colors
            # If successful, we reduce the color count by 1
            
            # First, try simple single-position changes that might enable color reduction
            for pos in range(n_local):
                if time.time() > deadline - 3:
                    break
                
                current_color = arr[pos]
                
                # Try reassigning to each other used color
                for new_color in used_colors:
                    if new_color == current_color:
                        continue
                    
                    if validate_change(arr, pos, current_color, new_color):
                        # Check if this reduces color count
                        old_freq = color_freq[current_color]
                        if old_freq == 1:
                            # This is the last position with this color - removing it reduces count
                            arr[pos] = new_color
                            color_freq[current_color] -= 1
                            del color_freq[current_color]
                            color_freq[new_color] = color_freq.get(new_color, 0) + 1
                            used_colors.remove(current_color)
                            current_colors -= 1
                            improved = True
                            break
                        else:
                            # Just a reassignment, doesn't reduce count directly
                            # But might enable future reductions - do it anyway if it doesn't hurt
                            arr[pos] = new_color
                            color_freq[current_color] -= 1
                            color_freq[new_color] = color_freq.get(new_color, 0) + 1
                            if color_freq[current_color] == 0:
                                del color_freq[current_color]
                                used_colors.remove(current_color)
                                current_colors -= 1
                                improved = True
                                break
                            else:
                                # Revert - this doesn't help immediately
                                arr[pos] = current_color
                                color_freq[current_color] += 1
                                color_freq[new_color] -= 1
                                if color_freq[new_color] == 0:
                                    del color_freq[new_color]
                                    used_colors.append(new_color)
                
                if improved:
                    break
            
            if not improved:
                # Try more aggressive: pick a color with fewest occurrences and try to eliminate it
                if len(used_colors) > 1:
                    # Find color with minimum frequency
                    min_color = min(used_colors, key=lambda c: color_freq[c])
                    min_freq = color_freq[min_color]
                    
                    # Try to reassign all positions with min_color to other colors
                    positions_with_min = [p for p in range(n_local) if arr[p] == min_color]
                    
                    # Try different orderings
                    random.shuffle(positions_with_min)
                    
                    success = True
                    new_arr = list(arr)
                    new_color_freq = dict(color_freq)
                    new_used = list(used_colors)
                    
                    for pos in positions_with_min:
                        placed = False
                        for new_color in new_used:
                            if new_color == min_color:
                                continue
                            if validate_change(new_arr, pos, min_color, new_color):
                                new_arr[pos] = new_color
                                new_color_freq[min_color] -= 1
                                new_color_freq[new_color] = new_color_freq.get(new_color, 0) + 1
                                placed = True
                                break
                        if not placed:
                            success = False
                            break
                    
                    if success and new_color_freq[min_color] == 0:
                        arr = new_arr
                        color_freq = new_color_freq
                        used_colors = new_used
                        used_colors.remove(min_color)
                        current_colors -= 1
                        improved = True
            
            if not improved:
                break
        
        return arr, current_colors
    
    # Run local search
    if time.time() < deadline - 10:
        improved_arr, improved_count = local_search(colors, max_iterations=100000)
        if improved_count < best_count:
            best_colors = improved_arr
            best_count = improved_count
    
    # Try random restarts with local search
    if time.time() < deadline - 5:
        random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed version of the best solution
            trial = list(best_colors)
            
            # Randomly perturb some positions
            num_perturb = random.randint(5, 15)
            for _ in range(num_perturb):
                pos = random.randint(0, n - 1)
                trial[pos] = random.randint(0, n - 1)
            
            # Try to fix and improve
            trial_arr, trial_count = local_search(trial, max_iterations=50000)
            
            if trial_count < best_count:
                best_colors = trial_arr
                best_count = trial_count
    
    # Final validation
    if not validate_full(best_colors):
        # Should not happen, but fallback
        best_colors = list(range(n))
    
    return best_colors

# EVOLVE-BLOCK-END
