"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
import numpy as np

def _precompute_progressions(n=128):
    """Precompute all nonconstant 4-term progressions as index tuples."""
    progs = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            e = a + 3*d
            if e < n:
                progs.append((a, a+d, a+2*d, e))
    return progs

def _check_valid_fast(colors, progs):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    for (a, b, c, e) in progs:
        ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
        # Count distinct colors - need at least 3
        # Fast rejection: if all same
        if ca == cb and ca == cc and ca == ce:
            return False
        # Check for exactly 2 distinct colors (invalid)
        # There are C(4,2)=6 ways to have pairs equal
        if ca == cb:
            # {ca, cc, ce} must have at least 3 distinct -> cc!=ca, ce!=ca, cc!=ce
            if cc == ca or ce == ca or cc == ce:
                return False
        elif ca == cc:
            if cb == ca or ce == ca or cb == ce:
                return False
        elif ca == ce:
            if cb == ca or cc == ca or cb == cc:
                return False
        elif cb == cc:
            if ca == cb or ce == cb or ca == ce:
                return False
        elif cb == ce:
            if ca == cb or cc == cb or ca == cc:
                return False
        elif cc == ce:
            if ca == cc or cb == cc or ca == cb:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_construct(max_colors, n=128, color_order=None):
    """Greedy construction."""
    if color_order is None:
        color_order = list(range(max_colors))
    
    colors = [-1] * n
    colors[0] = color_order[0]
    
    for x in range(1, n):
        found = False
        for color in color_order:
            colors[x] = color
            valid = True
            for d in range(1, x // 3 + 1):
                a = x - 3*d
                if a < 0:
                    break
                prog = {colors[a], colors[a+d], colors[a+2*d], colors[x]}
                if len(prog) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def _try_merge(colors, progs):
    """Try to merge two colors. Returns (merged_colors, success)."""
    unique = sorted(set(colors))
    if len(unique) <= 1:
        return colors, False
    
    for i in range(len(unique)):
        for j in range(i+1, len(unique)):
            trial = [unique[i] if c == unique[j] else c for c in colors]
            if _check_valid_fast(trial, progs):
                return trial, True
    return colors, False

def _local_search(colors, progs, deadline, max_rounds=50):
    """Local search to minimize colors."""
    colors = list(colors)
    best = list(colors)
    best_count = _count_colors(best)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if best_count <= 3:
            break
        
        # Try direct merge
        merged, success = _try_merge(best, progs)
        if success:
            best = merged
            best_count = _count_colors(best)
            continue
        
        # Try swap-based improvement
        improved = False
        n = len(best)
        for _ in range(500):
            if time.time() > deadline:
                break
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            trial = list(best)
            trial[i], trial[j] = trial[j], trial[i]
            
            if not _check_valid_fast(trial, progs):
                continue
            
            # Try to merge in trial
            t_merged, t_success = _try_merge(trial, progs)
            if t_success:
                best = t_merged
                best_count = _count_colors(best)
                improved = True
                break
        
        if not improved:
            # Try single position color change
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, n-1)
                new_color = random.randint(0, best_count - 1)
                if new_color == best[i]:
                    continue
                
                trial = list(best)
                trial[i] = new_color
                
                if not _check_valid_fast(trial, progs):
                    continue
                
                t_merged, t_success = _try_merge(trial, progs)
                if t_success:
                    best = t_merged
                    best_count = _count_colors(best)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    progs = _precompute_progressions(128)
    
    best_result = None
    best_count = 128
    
    # Try greedy with different max_colors and orderings
    for max_colors in range(3, 7):
        if time.time() > deadline - 5:
            break
        
        attempts = 10 if max_colors == 3 else 5
        for attempt in range(attempts):
            if time.time() > deadline - 3:
                break
            
            if attempt == 0:
                color_order = list(range(max_colors))
            else:
                color_order = list(range(max_colors))
                random.shuffle(color_order)
            
            colors = _greedy_construct(max_colors, 128, color_order=color_order)
            if colors is None:
                continue
            
            if not _check_valid_fast(colors, progs):
                continue
            
            unique = sorted(set(colors))
            cmap = {c: i for i, c in enumerate(unique)}
            remapped = [cmap[c] for c in colors]
            cnt = len(unique)
            
            if cnt < best_count:
                best_count = cnt
                best_result = remapped
            
            if best_count == 3:
                break
        
        if best_count == 3:
            break
    
    if best_result is None:
        # Fallback: simple assignment
        return list(range(128))
    
    # Local search to minimize
    if time.time() < deadline - 5:
        optimized = _local_search(best_result, progs, deadline - 2)
        if _check_valid_fast(optimized, progs):
            cnt = _count_colors(optimized)
            if cnt < best_count:
                best_result = optimized
                best_count = cnt
    
    return best_result
# EVOLVE-BLOCK-END
