import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    
    # Precompute 4-term APs ending at each position
    aps_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 0:
                aps_ending[i].append((a, d))
    
    # Precompute 3-term APs ending at each position
    threes_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 2 + 1):
            a = i - 2 * d
            if a >= 0:
                threes_ending[i].append((a, d))
    
    # Precompute all 4-term APs for verification
    all_aps = []
    for d in range(1, n // 3 + 1):
        for i in range(n - 3 * d):
            all_aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    def is_valid(colors):
        """Check if coloring is valid: every 4-term AP has >= 3 distinct colors."""
        for a, b, c, d in all_aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    def count_violations(colors):
        """Count number of 4-term APs with < 3 distinct colors."""
        count = 0
        for a, b, c, d in all_aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                count += 1
        return count
    
    def greedy_with_lookahead(max_color, randomize=False, seed=None):
        """
        Deterministic/randomized greedy: assign valid color at each position.
        Lookahead prevents 3-term APs from forcing a 2-color situation in the 4th term.
        """
        if seed is not None:
            random.seed(seed)
        
        colors = [None] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(max_color):
                # Check all 4-term APs ending at i
                ok_4 = True
                for a, d in aps_ending[i]:
                    if colors[a] is None or colors[a+d] is None or colors[a+2*d] is None:
                        continue
                    vals = {colors[a], colors[a+d], colors[a+2*d], c}
                    if len(vals) < 3:
                        ok_4 = False
                        break
                if not ok_4:
                    continue
                
                # Lookahead: for 3-term APs ending at i, ensure we don't force
                # the 4th term into a 2-color situation
                ok_3 = True
                for a, d in threes_ending[i]:
                    if colors[a] is None or colors[a+d] is None:
                        continue
                    # If colors[a] == c AND colors[a+d] == c, then the 4th term
                    # would need to be different from both, but might be forced
                    # by other constraints. The seed's condition: not both equal.
                    if colors[a] == c and colors[a+d] == c:
                        ok_3 = False
                        break
                if not ok_3:
                    continue
                
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
    
    def local_search(colors, max_iters=200):
        """
        Local search: try to reduce the number of colors used while maintaining validity.
        """
        colors = list(colors)
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            current_colors = set(colors)
        else:
            return None
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            current_colors = set(colors)
            
            # Try to recolor a position to an unused color or a color that reduces total colors
            improved = False
            for pos in range(n):
                if time.time() > deadline:
                    break
                
                old_color = colors[pos]
                
                # Try each color in the current palette
                for new_color in sorted(current_colors):
                    if new_color == old_color:
                        continue
                    
                    colors[pos] = new_color
                    v = count_violations(colors)
                    
                    if v == 0:
                        new_colors = set(colors)
                        if len(new_colors) < len(current_colors):
                            current_colors = new_colors
                            improved = True
                        else:
                            colors[pos] = old_color
                    else:
                        colors[pos] = old_color
                
                if improved:
                    break
            
            if not improved:
                break
        
        if is_valid(colors):
            return colors
        return None
    
    best_result = None
    best_num_colors = n + 1
    
    # Strategy 1: Deterministic greedy with full palette
    result = greedy_with_lookahead(128, randomize=False)
    if result is not None:
        num_used = len(set(result))
        if num_used < best_num_colors:
            best_num_colors = num_used
            best_result = result
    
    # Strategy 2: Try smaller palettes with randomized greedy
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            result = greedy_with_lookahead(k, randomize=True, seed=attempt)
            if result is not None:
                num_used = len(set(result))
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_result = result
    
    # Strategy 3: Local search refinement on best result
    if best_result is not None and time.time() < deadline - 10:
        refined = local_search(best_result, max_iters=100)
        if refined is not None:
            num_used = len(set(refined))
            if num_used < best_num_colors:
                best_num_colors = num_used
                best_result = refined
    
    if best_result is not None:
        return [int(x) for x in best_result]
    else:
        return list(range(n))
