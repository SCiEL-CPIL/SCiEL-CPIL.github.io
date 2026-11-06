"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _count_violations(colors, n=128):
    """Count four-term progressions that use fewer than 3 distinct colors."""
    # colors is 0-indexed array of length n
    c = colors
    violations = 0
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            a1 = c[a]
            a2 = c[a + d]
            a3 = c[a + 2 * d]
            a4 = c[a + 3 * d]
            # Count distinct colors
            if a1 == a2 == a3 == a4:
                violations += 1
            elif a1 == a2 == a3 or a1 == a2 == a4 or a1 == a3 == a4 or a2 == a3 == a4:
                violations += 1
            elif a1 == a2 and a3 == a4 and a1 != a3:
                violations += 1
            elif a1 == a3 and a2 == a4 and a1 != a2:
                violations += 1
            elif a1 == a4 and a2 == a3 and a1 != a2:
                violations += 1
    return violations

def _is_valid(colors, n=128):
    return _count_violations(colors, n) == 0

def _num_colors_used(colors):
    return len(set(colors.tolist()))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    n = 128
    best_colors = None
    best_score = float('inf')
    
    # Try greedy construction with different strategies
    # Strategy: assign colors greedily, preferring to reuse existing colors
    
    def greedy_construct(max_colors, seed=0):
        rng = random.Random(seed)
        colors = [0] * n
        used_colors = {0}
        
        for x in range(1, n):
            # Try to assign a color that minimizes violations
            best_color = None
            best_local_score = float('inf')
            
            # Candidate colors: existing used colors first, then new ones
            candidates = list(used_colors)
            if len(used_colors) < max_colors:
                candidates.append(len(used_colors))
            
            for color in candidates:
                colors[x] = color
                # Check if this creates any violations
                local_viol = 0
                # Check progressions ending at x
                for d in range(1, (x) // 3 + 1):
                    if x - 3 * d < 0:
                        break
                    a = x - 3 * d
                    vals = [colors[a], colors[a + d], colors[a + 2 * d], colors[a + 3 * d]]
                    distinct = len(set(vals))
                    if distinct < 3:
                        local_viol += 1
                
                # Also check progressions where x is not the last element
                # but is part of a 4-term AP
                # x = a, x = a+d, x = a+2d, x = a+3d
                for d in range(1, n // 3 + 1):
                    # x as first element
                    if x + 3 * d < n:
                        vals = [colors[x], colors[x + d], colors[x + 2 * d], colors[x + 3 * d]]
                        # Only check if all are assigned (they are, since we're filling left to right)
                        # But we only care about progressions that include x
                        distinct = len(set(vals))
                        if distinct < 3:
                            local_viol += 1
                    # x as second element
                    if x - d >= 0 and x + 2 * d < n:
                        a = x - d
                        vals = [colors[a], colors[a + d], colors[a + 2 * d], colors[a + 3 * d]]
                        distinct = len(set(vals))
                        if distinct < 3:
                            local_viol += 1
                    # x as third element
                    if x - 2 * d >= 0 and x + d < n:
                        a = x - 2 * d
                        vals = [colors[a], colors[a + d], colors[a + 2 * d], colors[a + 3 * d]]
                        distinct = len(set(vals))
                        if distinct < 3:
                            local_viol += 1
                
                if local_viol < best_local_score:
                    best_local_score = local_viol
                    best_color = color
                    if local_viol == 0:
                        break
                elif local_viol == best_local_score and best_color is not None:
                    # Prefer reusing existing colors
                    if color in used_colors and best_color not in used_colors:
                        best_color = color
            
            colors[x] = best_color
            used_colors.add(best_color)
        
        return colors
    
    # Try different numbers of colors
    for max_c in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            colors = greedy_construct(max_c, seed)
            score = _num_colors_used(colors)
            if _is_valid(colors):
                if score < best_score:
                    best_score = score
                    best_colors = colors
                if best_score == 3:  # Can't do better than 3
                    return best_colors
    
    # If no valid coloring found, try local search
    if best_colors is None or time.time() < deadline:
        # Start from a random coloring and try to reduce violations
        rng = random.Random(42)
        
        for init_seed in range(5):
            if time.time() > deadline:
                break
            
            colors = np.array([rng.randint(0, 4) for _ in range(n)], dtype=np.int32)
            current_viol = _count_violations(colors)
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                if current_viol == 0:
                    score = _num_colors_used(colors)
                    if score < best_score:
                        best_score = score
                        best_colors = colors
                    if best_score == 3:
                        return best_colors
                    break
                
                # Try flipping a random position
                idx = rng.randint(0, n - 1)
                old_color = colors[idx]
                new_color = rng.randint(0, 4)
                if new_color == old_color:
                    continue
                
                colors[idx] = new_color
                new_viol = _count_violations(colors)
                
                if new_viol <= current_viol:
                    current_viol = new_viol
                else:
                    colors[idx] = old_color
            
            # After local search, try to reduce number of colors
            if current_viol == 0 and best_colors is not None:
                # Try to merge color classes
                for merge_seed in range(3):
                    if time.time() > deadline:
                        break
                    
                    colors_copy = colors.copy()
                    used = list(set(colors_copy.tolist()))
                    rng2 = random.Random(merge_seed)
                    rng2.shuffle(used)
                    
                    for i in range(len(used)):
                        for j in range(i + 1, len(used)):
                            if time.time() > deadline:
                                break
                            # Try merging color j into color i
                            temp = colors_copy.copy()
                            temp[temp == used[j]] = used[i]
                            if _count_violations(temp) == 0:
                                colors_copy = temp
                                used = list(set(used))
                                used.remove(used[j])
            
            if current_viol == 0:
                score = _num_colors_used(colors)
                if score < best_score:
                    best_score = score
                    best_colors = colors
    
    if best_colors is not None:
        return best_colors.tolist()
    
    # Fallback: return a simple coloring (may not be valid, but must return something)
    return [i % 3 for i in range(n)]

# EVOLVE-BLOCK-END
