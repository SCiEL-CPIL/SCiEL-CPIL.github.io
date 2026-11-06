import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs
    all_aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d < n:
                all_aps.append((a, d))
    
    num_aps = len(all_aps)
    
    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(n)]
    for idx, (a, d) in enumerate(all_aps):
        for role in range(4):
            pos = a + role * d
            pos_aps[pos].append((idx, role))
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for a, d in all_aps:
            vals = {colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]}
            if len(vals) < 3:
                return False
        return True
    
    def try_merge(colors, color_a, color_b):
        """Try to merge color_b into color_a. Returns new colors if valid, else None."""
        new_colors = list(colors)
        changed = False
        for i in range(n):
            if new_colors[i] == color_b:
                new_colors[i] = color_a
                changed = True
        if not changed:
            return None
        
        # Check validity
        for a, d in all_aps:
            vals = {new_colors[a], new_colors[a+d], new_colors[a+2*d], new_colors[a+3*d]}
            if len(vals) < 3:
                return None
        return new_colors
    
    def merge_pass(colors, max_iterations=300):
        """Iteratively try to merge color pairs until no more merges possible."""
        current = list(colors)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            used_colors = sorted(set(current))
            if len(used_colors) <= 3:
                break
            merged = False
            for i in range(len(used_colors)):
                for j in range(i+1, len(used_colors)):
                    if time.time() > deadline:
                        return current
                    ca, cb = used_colors[i], used_colors[j]
                    result = try_merge(current, ca, cb)
                    if result is not None:
                        current = result
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        return current
    
    def greedy_construct(k_colors, randomize=True):
        """Construct a valid coloring using k colors via greedy."""
        colors = [-1] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(k_colors):
                ok = True
                for ap_idx, role in pos_aps[i]:
                    a, d = all_aps[ap_idx]
                    vals = set()
                    for r in range(4):
                        pos = a + r * d
                        if r == role:
                            vals.add(c)
                        else:
                            vals.add(colors[pos])
                    if len(vals) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            if randomize:
                colors[i] = random.choice(valid_colors)
            else:
                colors[i] = valid_colors[0]
        
        if is_valid(colors):
            return colors
        return None
    
    # Strategy: Use modular construction as seed, then improve
    # Try different moduli as starting points
    best_colors = None
    best_count = n + 1
    
    # Try modular seeds for m = 4 to 10
    for m in range(4, 11):
        if time.time() > deadline:
            break
        colors = [i % m for i in range(n)]
        if is_valid(colors):
            cnt = count_colors(colors)
            if cnt < best_count:
                best_colors = list(colors)
                best_count = cnt
            merged = merge_pass(colors)
            mcnt = count_colors(merged)
            if mcnt < best_count:
                best_colors = merged
                best_count = mcnt
    
    # Try greedy with various color counts
    for k in [6, 7, 8, 10, 12, 16]:
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            result = greedy_construct(k, randomize=True)
            if result is not None:
                cnt = count_colors(result)
                if cnt < best_count:
                    best_colors = list(result)
                    best_count = cnt
                merged = merge_pass(result)
                mcnt = count_colors(merged)
                if mcnt < best_count:
                    best_colors = merged
                    best_count = mcnt
    
    # If no valid coloring found, use fallback
    if best_colors is None:
        # Fallback: all distinct
        best_colors = list(range(n))
    else:
        # Local search: try to reduce colors by changing individual positions
        current_best = list(best_colors)
        current_best_count = best_count
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random position and try changing its color to an existing color
            pos = random.randint(0, n - 1)
            old_color = current_best[pos]
            
            # Try merging this position's color into another existing color
            used = list(set(current_best))
            random.shuffle(used)
            
            improved = False
            for new_color in used:
                if new_color == old_color:
                    continue
                
                current_best[pos] = new_color
                
                # Check validity
                valid = True
                for ap_idx, role in pos_aps[pos]:
                    a, d = all_aps[ap_idx]
                    vals = {current_best[a], current_best[a+d], current_best[a+2*d], current_best[a+3*d]}
                    if len(vals) < 3:
                        valid = False
                        break
                
                if valid:
                    new_count = count_colors(current_best)
                    if new_count < current_best_count:
                        current_best_count = new_count
                        improved = True
                        break
                    else:
                        # Revert if not reducing color count
                        current_best[pos] = old_color
                else:
                    current_best[pos] = old_color
            
            if not improved and iteration % 100 == 0:
                # Occasionally try a full merge pass
                merged = merge_pass(current_best)
                mcnt = count_colors(merged)
                if mcnt < current_best_count:
                    current_best = merged
                    current_best_count = mcnt
        
        if current_best_count < best_count:
            best_colors = current_best
            best_count = current_best_count
    
    # Final merge pass
    if best_colors is not None:
        final = merge_pass(best_colors)
        if count_colors(final) < best_count:
            best_colors = final
    
    return best_colors
