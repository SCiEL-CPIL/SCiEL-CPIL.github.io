"""Rewritten program with stronger local search for 4-term progression coloring."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions
    progressions = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j3 = i + 3 * d
            if j3 < n:
                progressions.append((i, i + d, i + 2 * d, j3))
    
    def validate(arr):
        """Check if every nonconstant 4-term progression uses at least 3 distinct colors."""
        for (a, b, c, d) in progressions:
            colors = {arr[a], arr[b], arr[c], arr[d]}
            if len(colors) < 3:
                return False
        return True
    
    def count_colors(arr):
        return len(set(arr))
    
    def local_search(arr, max_iterations=200000, deadline=deadline):
        """Improve coloring by trying to eliminate colors."""
        arr = list(arr)
        n = len(arr)
        current_colors = count_colors(arr)
        
        # Build adjacency: for each position, which progressions involve it
        pos_progressions = [[] for _ in range(n)]
        for idx, (a, b, c, d) in enumerate(progressions):
            pos_progressions[a].append(idx)
            pos_progressions[b].append(idx)
            pos_progressions[c].append(idx)
            pos_progressions[d].append(idx)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            improved = False
            used_colors = sorted(set(arr))
            
            for color_to_remove in used_colors:
                if time.time() > deadline - 2:
                    break
                
                # Find all positions with this color
                positions_with_color = [i for i in range(n) if arr[i] == color_to_remove]
                
                if not positions_with_color:
                    continue
                
                # Try to reassign each position with this color to another color
                trial = list(arr)
                success = True
                
                for pos in positions_with_color:
                    # Try each other color
                    found_valid = False
                    for new_color in used_colors:
                        if new_color == color_to_remove:
                            continue
                        
                        trial[pos] = new_color
                        # Check all progressions involving pos
                        valid = True
                        for pidx in pos_progressions[pos]:
                            a, b, c, d = progressions[pidx]
                            colors = {trial[a], trial[b], trial[c], trial[d]}
                            if len(colors) < 3:
                                valid = False
                                break
                        
                        if valid:
                            found_valid = True
                            break
                    
                    if not found_valid:
                        success = False
                        break
                
                if success:
                    # Verify full validity
                    if validate(trial):
                        arr = trial
                        current_colors -= 1
                        improved = True
                        break
            
            if not improved:
                break
        
        return arr, current_colors
    
    def greedy_construct(max_colors=128):
        """Build a valid coloring greedily, trying to use few colors."""
        colors = [-1] * n
        
        for x in range(n):
            # Try colors in order of least use
            color_counts = {}
            for c in range(max_colors):
                color_counts[c] = sum(1 for i in range(x) if colors[i] == c)
            
            # Sort colors by usage count (try least used first)
            sorted_colors = sorted(range(max_colors), key=lambda c: color_counts[c])
            
            for color in sorted_colors:
                # Check all progressions ending at x
                valid = True
                for (a, b, c, d) in progressions:
                    if d == x:
                        # Check if this progression would be valid with color at x
                        existing_colors = {colors[a], colors[b], colors[c]}
                        if -1 in existing_colors:
                            continue  # Can't check yet
                        all_colors = existing_colors | {color}
                        if len(all_colors) < 3:
                            valid = False
                            break
                    elif a == x:
                        # x is the start
                        existing_colors = {colors[b], colors[c], colors[d]}
                        if -1 in existing_colors:
                            continue
                        all_colors = existing_colors | {color}
                        if len(all_colors) < 3:
                            valid = False
                            break
                    elif b == x:
                        existing_colors = {colors[a], colors[c], colors[d]}
                        if -1 in existing_colors:
                            continue
                        all_colors = existing_colors | {color}
                        if len(all_colors) < 3:
                            valid = False
                            break
                    elif c == x:
                        existing_colors = {colors[a], colors[b], colors[d]}
                        if -1 in existing_colors:
                            continue
                        all_colors = existing_colors | {color}
                        if len(all_colors) < 3:
                            valid = False
                            break
                
                if valid:
                    colors[x] = color
                    break
            
            if colors[x] == -1:
                # Fallback: use a new color
                colors[x] = max_colors
                max_colors += 1
        
        return colors
    
    # Start with greedy construction
    best_colors = greedy_construct()
    best_count = count_colors(best_colors)
    
    # Ensure validity
    if not validate(best_colors):
        # Fallback: use a simple pattern
        best_colors = list(range(n))
        best_count = n
    
    # Local search to reduce colors
    if time.time() < deadline - 5:
        improved_arr, improved_count = local_search(best_colors, max_iterations=100000)
        if improved_count < best_count:
            best_colors = improved_arr
            best_count = improved_count
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 3:
            break
        
        # Create a random starting point
        num_colors = random.randint(3, min(8, best_count))
        trial = [random.randint(0, num_colors - 1) for _ in range(n)]
        
        # Try to fix invalidities
        for fix_iter in range(200):
            if time.time() > deadline - 3:
                break
            
            # Find first invalid progression
            invalid_found = False
            for (a, b, c, d) in progressions:
                colors = {trial[a], trial[b], trial[c], trial[d]}
                if len(colors) < 3:
                    # Try to fix by changing one position
                    for pos in [a, b, c, d]:
                        for new_color in range(0, num_colors + 1):
                            if new_color == trial[pos]:
                                continue
                            trial[pos] = new_color
                            # Check if this fixes the progression
                            colors = {trial[a], trial[b], trial[c], trial[d]}
                            if len(colors) >= 3:
                                invalid_found = True
                                break
                            trial[pos] = new_color  # revert
                        if invalid_found:
                            break
                    if invalid_found:
                        break
            
            if not invalid_found:
                break
        
        if validate(trial):
            trial_count = count_colors(trial)
            if trial_count < best_count:
                # Try local search to improve further
                improved_arr, improved_count = local_search(trial, max_iterations=50000)
                if improved_count < best_count:
                    best_colors = improved_arr
                    best_count = improved_count
                elif trial_count < best_count:
                    best_colors = trial
                    best_count = trial_count
    
    return best_colors
