import random
import time
import sys

def _precompute_progressions(n):
    """Precompute all nonconstant 4-term progressions as (a, b, c, d) tuples of 0-indexed positions."""
    progs = []
    # 4-term AP: a, a+d, a+2d, a+3d with d >= 1
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            progs.append((a, a + d, a + 2 * d, a + 3 * d))
    return progs

def _check_validity(colors, progs):
    """Check if every progression uses at least 3 distinct colors."""
    for (a, b, c, d) in progs:
        ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
        # Count distinct
        if ca == cb == cc == cd:
            return False
        if ca == cb and ca == cc and ca == cd:
            return False
        if ca == cb and ca == cd and cb == cc:
            return False
        if ca == cc and ca == cd and cb == cc:
            return False
        if ca == cd and cb == cc and cb == cd:
            return False
        # Check if only 2 distinct colors
        s = {ca, cb, cc, cd}
        if len(s) < 3:
            return False
    return True

def _count_violations(colors, progs):
    """Count number of progressions that violate the constraint (use < 3 colors)."""
    violations = 0
    for (a, b, c, d) in progs:
        ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
        s = {ca, cb, cc, cd}
        if len(s) < 3:
            violations += 1
    return violations

def _num_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline with 10s safety margin
    
    n = 128
    progs = _precompute_progressions(n)
    
    # Strategy: Start with a random 3-coloring, verify it's valid,
    # then use local search to try to reduce to fewer colors.
    
    best_colors = None
    best_score = 128  # max possible colors
    
    # First, find a valid 3-coloring via random restarts
    rng = random.Random(42)
    
    # Try to find a valid 3-coloring
    valid_3color = None
    for attempt in range(5000):
        if time.time() > deadline - 10:
            break
        colors = [rng.randint(0, 2) for _ in range(n)]
        if _check_validity(colors, progs):
            valid_3color = colors[:]
            break
    
    if valid_3color is None:
        # Fallback: greedy construction with 3 colors
        colors = [-1] * n
        for i in range(n):
            for c in range(3):
                colors[i] = c
                ok = True
                for (a, b, cd, d) in progs:
                    if cd == i or d == i:
                        if all(colors[x] >= 0 for x in (a, b, cd, d)):
                            s = {colors[a], colors[b], colors[cd], colors[d]}
                            if len(s) < 3:
                                ok = False
                                break
                if ok:
                    break
            if colors[i] < 0:
                colors[i] = 0  # force
        if _check_validity(colors, progs):
            valid_3color = colors[:]
    
    if valid_3color is not None:
        best_colors = valid_3color[:]
        best_score = _num_colors(valid_3color)
    
    # Now try local search to reduce colors
    # We want to try to use only 2 colors, or even 1 (impossible for 4-APs)
    # Actually with 2 colors, a 4-term AP can use both colors (2 distinct) which is < 3, so invalid.
    # So minimum is 3 colors.
    
    # Since minimum is 3, and we found a valid 3-coloring, score = 3.
    # But wait - can we do better? The constraint says AT LEAST 3 distinct colors.
    # So we need exactly 3 or more. Minimum is 3.
    
    # Let's verify: can 3 colors work? Yes, we found one.
    # Can 2 colors work? No, because any 4-term AP with 2 colors uses at most 2 distinct.
    # So the answer is 3.
    
    # But we need to make sure our 3-coloring is truly valid.
    # Let's also try to find one more robustly.
    
    if best_colors is None:
        # More aggressive search for valid 3-coloring
        for attempt in range(100000):
            if time.time() > deadline - 5:
                break
            colors = [rng.randint(0, 2) for _ in range(n)]
            if _check_validity(colors, progs):
                best_colors = colors[:]
                best_score = 3
                break
    
    if best_colors is None:
        # Last resort: use the greedy approach from the seed
        colors = []
        for x in range(1, 129):
            placed = False
            for color in range(3):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    j1, j2, j3 = x - 1 - d, x - 1 - 2 * d, x - 1 - 3 * d
                    if j1 >= 0 and j2 >= 0 and j3 >= 0:
                        s = {color, colors[j1], colors[j2], colors[j3]}
                        if len(s) < 3:
                            ok = False
                            break
                if ok:
                    colors.append(color)
                    placed = True
                    break
            if not placed:
                colors.append(0)
        best_colors = colors
        best_score = _num_colors(colors)
    
    # Return with labels in 0..127
    # Remap colors to 0, 1, 2
    unique = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique)}
    result = [mapping[c] for c in best_colors]
    
    # Final validation
    assert len(result) == 128
    assert all(0 <= x <= 127 for x in result)
    assert _check_validity(result, progs), "Final result is not valid!"
    
    return result

# Precompute at module level for speed
_PROGS = _precompute_progressions(128)

# Override _check_validity to use precomputed
def _check_validity(colors, progs=None):
    if progs is None:
        progs = _PROGS
    for (a, b, c, d) in progs:
        ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
        s = {ca, cb, cc, cd}
        if len(s) < 3:
            return False
    return True
