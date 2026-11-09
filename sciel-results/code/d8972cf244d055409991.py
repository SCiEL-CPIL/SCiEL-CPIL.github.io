"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
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
    
    def is_valid_change(arr, pos, new_color):
        """Check if setting arr[pos] = new_color maintains validity.
        Only checks progressions containing pos."""
        for (pi, others) in pos_progs[pos]:
            prog_colors = {new_color}
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
                max_d_3rd = pos // 2
                for d in range(1, max_d_3rd + 1):
                    i = pos - 2 * d
                    if i < 0:
                        continue
                    c0 = colors[i]
                    c1 = colors[i + d]
                    c2 = color
                    if c0 == c1 and c1 == c2:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors.append(color)
                found = True
                break
            if not found:
                colors.append(0)
        return colors
    
    # Start with greedy construction
    colors = greedy_construct()
    
    # Verify it's valid
    if not validate_full(colors):
        colors = list(range(n))
    
    best_colors = colors[:]
    best_count = count_colors(colors)
    
    # Local search to reduce color count
    def local_search(arr, max_iterations=500000):
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
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # Strategy 1: Try to eliminate a color by reassigning all its positions
            if len(used_colors) > 1:
                # Find color with minimum frequency
                min_color = min(used_colors, key=lambda c: color_freq[c])
                min_freq = color_freq[min_color]
                positions_with_min = [p for p in range(n_local) if arr[p] == min_color]
                
                # Try different orderings (up to 3 random shuffles)
                for attempt in range(3):
                    random.shuffle(positions_with_min)
                    
                    success = True
                    new_arr = list(arr)
                    new_color_freq = dict(color_freq)
                    
                    for pos in positions_with_min:
                        placed = False
                        # Try other used colors in order of increasing frequency (less crowded first)
                        other_colors = [c for c in used_colors if c != min_color]
                        other_colors.sort(key=lambda c: new_color_freq.get(c, 0))
                        
                        for new_color in other_colors:
                            if is_valid_change(new_arr, pos, new_color):
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
                        used_colors.remove(min_color)
                        current_colors -= 1
                        improved = True
                        break
            
            # Strategy 2: Single position reassignment that eliminates a color
            if not improved:
                for pos in range(n_local):
                    if time.time() > deadline - 2:
                        break
                    
                    current_color = arr[pos]
                    old_freq = color_freq[current_color]
                    
                    if old_freq == 1:
                        # This is the only position with this color - try to reassign it
                        for new_color in used_colors:
                            if new_color == current_color:
                                continue
                            if is_valid_change(arr, pos, new_color):
                                arr[pos] = new_color
                                del color_freq[current_color]
                                color_freq[new_color] = color_freq.get(new_color, 0) + 1
                                used_colors.remove(current_color)
                                current_colors -= 1
                                improved = True
                                break
                    else:
                        # Try reassigning to see if it enables a future elimination
                        # Only do this if it doesn't increase the color count
                        for new_color in used_colors:
                            if new_color == current_color:
                                continue
                            if is_valid_change(arr, pos, new_color):
                                # Check if this creates a new singleton color
                                if color_freq[new_color] == 0:
                                    # Don't do this - it would increase color count
                                    continue
                                arr[pos] = new_color
                                color_freq[current_color] -= 1
                                color_freq[new_color] += 1
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
                
                if improved:
                    continue
            
            # Strategy 3: Simulated annealing step - accept some non-improving moves
            if not improved and iteration % 10 == 0:
                # Pick a random position and try a random color
                pos = random.randint(0, n_local - 1)
                current_color = arr[pos]
                
                # Try a random different color
                if len(used_colors) > 1:
                    new_color = random.choice([c for c in used_colors if c != current_color])
                    if is_valid_change(arr, pos, new_color):
                        # Accept with some probability (annealing)
                        if random.random() < 0.3:
                            arr[pos] = new_color
                            color_freq[current_color] -= 1
                            color_freq[new_color] = color_freq.get(new_color, 0) + 1
                            if color_freq[current_color] == 0:
                                del color_freq[current_color]
                                used_colors.remove(current_color)
                                current_colors -= 1
                            if new_color not in used_colors:
                                used_colors.append(new_color)
                                current_colors += 1
                            improved = True
            
            if not improved and iteration % 1000 == 999:
                # Check if we're stuck - try a more aggressive perturbation
                if len(used_colors) > 2:
                    # Pick two colors and try to swap some positions between them
                    c1, c2 = random.sample(used_colors, 2)
                    pos1 = [p for p in range(n_local) if arr[p] == c1]
                    pos2 = [p for p in range(n_local) if arr[p] == c2]
                    
                    if pos1 and pos2:
                        # Try swapping one position from each
                        p1 = random.choice(pos1)
                        p2 = random.choice(pos2)
                        
                        # Check if swapping is valid
                        temp = arr[p1]
                        arr[p1] = arr[p2]
                        arr[p2] = temp
                        
                        valid = True
                        for pos in (p1, p2):
                            for (pi, others) in pos_progs[pos]:
                                prog_colors = {arr[pos]}
                                for o in others:
                                    prog_colors.add(arr[o])
                                if len(prog_colors) < 3:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            # Update frequencies (swapping doesn't change total)
                            pass
                        else:
                            # Revert
                            arr[p2] = arr[p1]
                            arr[p1] = temp
            
            if not improved and iteration > 10000:
                # Stuck for a while - break out
                break
        
        return arr, current_colors
    
    # Run initial local search
    if time.time() < deadline - 15:
        improved_arr, improved_count = local_search(colors, max_iterations=200000)
        if improved_count < best_count:
            best_colors = improved_arr
            best_count = improved_count
    
    # Multiple restarts with perturbation
    if time.time() < deadline - 10:
        random.seed(42)
        for restart in range(15):
            if time.time() > deadline - 10:
                break
            
            # Create a perturbed version of the best solution
            trial = list(best_colors)
            
            # Randomly perturb some positions
            num_perturb = random.randint(3, 10)
            for _ in range(num_perturb):
                pos = random.randint(0, n - 1)
                # Assign a random color that's in the current palette or a new one
                if random.random() < 0.7:
                    trial[pos] = random.choice(set(best_colors))
                else:
                    trial[pos] = random.randint(0, n - 1)
            
            # Try to fix and improve
            trial_arr, trial_count = local_search(trial, max_iterations=100000)
            
            if trial_count < best_count:
                best_colors = trial_arr
                best_count = trial_count
    
    # Final polish pass with more iterations
    if time.time() < deadline - 5:
        final_arr, final_count = local_search(best_colors, max_iterations=300000)
        if final_count < best_count:
            best_colors = final_arr
            best_count = final_count
    
    # Final validation
    if not validate_full(best_colors):
        # Should not happen, but fallback
        best_colors = list(range(n))
    
    return best_colors

# EVOLVE-BLOCK-END
