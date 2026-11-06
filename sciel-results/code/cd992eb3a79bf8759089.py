"""Colorings with three colors in every four-term progression."""
import random
import time
import sys

# EVOLVE-BLOCK-START

def _precompute_ap4s():
    """Precompute all nonconstant 4-term APs (a, a+d, a+2d, a+3d) with a+3d <= 128.
    Return list of tuples (i0, i1, i2, i3) using 0-based indices."""
    ap4s = []
    for d in range(1, 128 // 3 + 1):
        max_a = 128 - 3 * d
        for a in range(1, max_a + 1):
            # 0-based: a-1, a-1+d, a-1+2d, a-1+3d
            ap4s.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    return ap4s

def _precompute_constraints():
    """For each position i (0-based), precompute:
    - ap4s_starting[i]: list of (i1, i2, i3) where i is the first element
    - ap4s_containing[i]: list of (i0, i1, i2, i3) where i is one of the four
    Actually, for greedy we need: when assigning position i, check all APs that
    have i as a member and where the other 3 positions are already assigned.
    
    Better approach: for each position i, precompute all APs where i is the LAST
    unassigned position (i.e., i is the max index in the AP). Then when we assign
    position i in order 0,1,2,...,127, we only need to check APs where i is the
    last position.
    
    Wait, that's not right either. Let me think differently.
    
    For greedy left-to-right: when assigning position i, all positions < i are
    already assigned. An AP (a, a+d, a+2d, a+3d) with 0-based indices has
    positions a, a+d, a+2d, a+3d. When we're at position i, this AP is fully
    determined (all 4 positions assigned) only if a+3d == i. But we also need
    to check APs where i is NOT the last position but some earlier position is
    unassigned... no, we're going left to right so all positions < i are assigned.
    
    So: when assigning position i, we need to check all APs where i is the
    maximum index in the AP. That means a+3d = i, so a = i-3d for various d.
    Also, i could be a+2d (then a+3d > i, not yet assigned) - no, we only check
    fully-assigned APs.
    
    Actually wait: the constraint is that EVERY nonconstant 4-term AP must use
    at least 3 distinct colors. When we assign position i, the only APs that
    become "complete" (all 4 positions assigned) are those where i is the last
    position (a+3d = i). For APs where i is not the last position, they'll be
    checked later.
    
    But we also need to ensure that the color we pick for i doesn't create a
    situation where a future AP will be forced to violate. That's the hard part
    of greedy - we need lookahead.
    
    Alternative: just check all APs where i is a member and the other 3 are
    already assigned. Since we go left to right, the other 3 are assigned iff
    they're all < i. For AP (a, a+d, a+2d, a+3d) with 0-based:
    - If i = a+3d: other 3 are a, a+d, a+2d, all < i. CHECK.
    - If i = a+2d: other 3 are a, a+d, a+3d. a+3d > i, NOT assigned. Don't check.
    - If i = a+d: other 3 include a+2d, a+3d > i. Don't check.
    - If i = a: other 3 all > i. Don't check.
    
    So indeed, only APs where i is the last position need checking at step i.
    """
    # For each position i (0-based), list of (i0, i1, i2) where i = i3 = i0+3d
    last_pos_ap4s = [[] for _ in range(128)]
    for d in range(1, 128 // 3 + 1):
        for a in range(1, 128 - 3 * d + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            last_pos_ap4s[i3].append((i0, i1, i2))
    return last_pos_ap4s

def _greedy_construct(palette_size, last_pos_ap4s, rng):
    """Greedy left-to-right construction with given palette.
    Returns list of 128 colors (0-based indices into palette) or None if stuck."""
    colors = [-1] * 128
    palette = list(range(palette_size))
    
    for i in range(128):
        # Determine which colors are allowed at position i
        allowed = set(palette)
        for (i0, i1, i2) in last_pos_ap4s[i]:
            c0, c1, c2 = colors[i0], colors[i1], colors[i2]
            # The AP (c0, c1, c2, ci) must have at least 3 distinct colors
            # So ci cannot make all 4 the same, and cannot make exactly 2 distinct
            # if the existing 3 already have exactly 2 distinct.
            existing = {c0, c1, c2}
            if len(existing) == 1:
                # All three same color. ci must be different to get 2 distinct... 
                # wait, we need AT LEAST 3 distinct. So if existing has 1 color,
                # ci can be anything (gives 2 distinct) - NO, that's only 2.
                # We need at least 3, so ci must NOT be in existing.
                # Actually: existing = {x}. If ci = x, total = {x} (1 distinct) - BAD.
                # If ci != x, total = {x, ci} (2 distinct) - still BAD (need >= 3).
                # So this is IMPOSSIBLE to satisfy! 
                # Wait, that means if three positions in an AP have the same color,
                # the fourth can't fix it. So we must avoid three same-color positions
                # in any AP.
                # This means: for any AP, no color appears 3+ times among the 4.
                # If existing 3 are all same, NO valid color for i. Return None.
                return None
            elif len(existing) == 2:
                # Two distinct colors, say {a, b}. 
                # If ci = a: total = {a, b} (2 distinct) - BAD
                # If ci = b: total = {a, b} (2 distinct) - BAD  
                # If ci = c (new): total = {a, b, c} (3 distinct) - OK
                # So ci must NOT be in existing.
                for c in existing:
                    allowed.discard(c)
            elif len(existing) == 3:
                # Three distinct colors. Any ci works (gives 3 or 4 distinct).
                pass
        
        if not allowed:
            return None
        
        # Pick a color from allowed (randomly for diversity)
        color_list = list(allowed)
        colors[i] = rng.choice(color_list)
    
    return colors

def _verify(colors):
    """Verify that every nonconstant 4-term AP uses at least 3 distinct colors."""
    for d in range(1, 128 // 3 + 1):
        for a in range(1, 128 - 3 * d + 1):
            c = {colors[a-1], colors[a-1+d], colors[a-1+2*d], colors[a-1+3*d]}
            if len(c) < 3:
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    last_pos_ap4s = _precompute_constraints()
    
    best_result = None
    best_num_colors = 128
    
    # Try palette sizes from 3 upward
    for palette_size in range(3, 13):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds for this palette size
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 7919 + palette_size * 31)
            result = _greedy_construct(palette_size, last_pos_ap4s, rng)
            
            if result is not None:
                if _verify(result):
                    num_used = len(set(result))
                    if num_used < best_num_colors:
                        best_num_colors = num_used
                        best_result = result[:]
                    # If we found a valid coloring with exactly palette_size colors,
                    # we can't do better for this palette, but maybe smaller palette
                    # works. Continue to try smaller... wait, we're going up.
                    # Actually we should try to minimize, so if we found one with
                    # palette_size, check if we can do better. But we started at 3,
                    # so if 3 works, that's optimal.
                    if num_used == palette_size:
                        # Found valid with this many colors
                        break
    
    if best_result is not None:
        # Map colors to 0..127 range (they already are 0..palette_size-1 which is in 0..127)
        return best_result
    
    # Fallback: use the seed program's approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
