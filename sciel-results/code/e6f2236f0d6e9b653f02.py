"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_progressions(n=128):
    """Precompute all nonconstant 4-term progressions as index tuples."""
    progs = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e < n:
                progs.append((a, a + d, a + 2 * d, e))
    return progs

PROGRESSIONS = _precompute_progressions()
PROG_ARRAY = np.array(PROGRESSIONS, dtype=np.int32)

def _check_valid_fast(colors):
    """Fast validation using numpy. Returns True if valid."""
    if len(colors) == 0:
        return True
    arr = np.array(colors, dtype=np.int32)
    # For each progression, check if it has at least 3 distinct colors
    # A progression is invalid if it has 1 or 2 distinct colors
    # 1 distinct: all 4 same
    # 2 distinct: exactly 2 colors used
    for i in range(len(PROG_ARRAY)):
        a, b, c, e = PROG_ARRAY[i]
        ca, cb, cc, ce = arr[a], arr[b], arr[c], arr[e]
        # Count distinct colors
        if ca == cb == cc == ce:
            return False
        # Check if exactly 2 colors
        if ca == cb and ca == cc and ca == ce:
            return False
        if ca == cb and ca == cc and cb != ce:
            return False
        if ca == cb and ca != cc and ca != ce:
            if cc == ce:
                return False
        if ca != cb and ca == cc and ca == ce:
            return False
        if ca != cb and ca != cc and ca == ce:
            if cb == cc:
                return False
        if ca != cb and ca != cc and ca != ce:
            if cb == cc and cb == ce:
                return False
            if cb == ce and cb != cc:
                return False
            if cc == ce and cc != cb:
                return False
    return True

def _check_valid(colors):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    n = len(colors)
    for a, b, c, e in PROGRESSIONS:
        prog = {colors[a], colors[b], colors[c], colors[e]}
        if len(prog) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_construct(max_colors=4):
    """Greedy construction trying to use at most max_colors colors."""
    n = 128
    colors = [0] * n
    # Precompute which progressions end at each position
    progs_ending_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(PROGRESSIONS):
        progs_ending_at[e].append(idx)
    
    for x in range(n):
        found = False
        for color in range(max_colors):
            colors[x] = color
            valid = True
            for idx in progs_ending_at[x]:
                a, b, c, e = PROGRESSIONS[idx]
                prog = {colors[a], colors[b], colors[c], colors[e]}
                if len(prog) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def _greedy_construct_random(max_colors=4, seed=None):
    """Greedy construction with randomized color ordering."""
    n = 128
    rng = random.Random(seed)
    colors = [0] * n
    progs_ending_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(PROGRESSIONS):
        progs_ending_at[e].append(idx)
    
    for x in range(n):
        color_order = list(range(max_colors))
        rng.shuffle(color_order)
        found = False
        for color in color_order:
            colors[x] = color
            valid = True
            for idx in progs_ending_at[x]:
                a, b, c, e = PROGRESSIONS[idx]
                prog = {colors[a], colors[b], colors[c], colors[e]}
                if len(prog) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors

def _local_search_reassign(colors, max_colors, deadline):
    """Local search that reassigns individual positions to minimize colors."""
    colors = list(colors)
    n = 128
    best = _count_colors(colors)
    current = best
    rng = random.Random()
    
    # Precompute progressions touching each position
    progs_at = [[] for _ in range(n)]
    for idx, (a, b, c, e) in enumerate(PROGRESSIONS):
        for pos in (a, b, c, e):
            progs_at[pos].append(idx)
    
    while time.time() < deadline and current > 1:
        improved = False
        # Try reassigning each position to each color
        positions = list(range(n))
        rng.shuffle(positions)
        for pos in positions:
            if time.time() > deadline:
                break
            old_color = colors[pos]
            # Try each color from 0 to max_colors-1
            for new_color in range(max_colors):
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                # Check only progressions touching this position
                valid = True
                for idx in progs_at[pos]:
                    a, b, c, e = PROGRESSIONS[idx]
                    prog = {colors[a], colors[b], colors[c], colors[e]}
                    if len(prog) < 3:
                        valid = False
                        break
                if valid:
                    new_count = _count_colors(colors)
                    if new_count < current:
                        current = new_count
                        improved = True
                        break
                    elif new_count == current:
                        # Accept with some probability for diversification
                        if rng.random() < 0.3:
                            improved = True
                            break
                    colors[pos] = old_color
            if improved:
                break
        if not improved:
            break
    
    return colors, current

def _merge_colors(colors, c1, c2):
    """Merge color c2 into c1."""
    return [c1 if c == c2 else c for c in colors]

def _local_search_merge(colors, deadline):
    """Local search that merges colors to reduce count."""
    colors = list(colors)
    unique_colors = sorted(set(colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    remapped = [color_map[c] for c in colors]
    best = _count_colors(remapped)
    
    while best > 1 and time.time() < deadline:
        improved = False
        for c1 in range(best):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best):
                new_colors = _merge_colors(remapped, c2, c1)
                if _check_valid(new_colors):
                    remapped = new_colors
                    best -= 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return remapped

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s safety margin
    
    best_colors = None
    best_count = 128
    
    # Phase 1: Try greedy with 3 colors (multiple random seeds)
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        colors = _greedy_construct_random(3, seed)
        if colors is not None and _check_valid(colors):
            count = _count_colors(colors)
            if count < best_count:
                best_count = count
                best_colors = colors
            if count <= 3:
                break
    
    # Phase 2: Try greedy with 4 colors
    if best_count > 4:
        for seed in range(20):
            if time.time() > deadline - 20:
                break
            colors = _greedy_construct_random(4, seed)
            if colors is not None and _check_valid(colors):
                count = _count_colors(colors)
                if count < best_count:
                    best_count = count
                    best_colors = colors
    
    # Phase 3: If we have a solution, try to optimize
    if best_colors is not None:
        # Try merge-based local search
        if time.time() < deadline - 10:
            merged = _local_search_merge(best_colors, deadline - 5)
            if _check_valid(merged):
                count = _count_colors(merged)
                if count < best_count:
                    best_count = count
                    best_colors = merged
        
        # Try reassignment-based local search
        if time.time() < deadline - 5:
            reassigned, new_count = _local_search_reassign(best_colors, best_count, deadline - 1)
            if _check_valid(reassigned) and new_count < best_count:
                best_count = new_count
                best_colors = reassigned
    
    # Phase 4: If no solution found, use fallback greedy with more colors
    if best_colors is None:
        for max_colors in range(5, 128):
            if time.time() > deadline - 10:
                break
            colors = _greedy_construct(max_colors)
            if colors is not None and _check_valid(colors):
                best_colors = colors
                best_count = _count_colors(colors)
                break
    
    # Emergency fallback
    if best_colors is None:
        best_colors = list(range(128))
    
    return best_colors
# EVOLVE-BLOCK-END
