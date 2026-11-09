"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _precompute_aps(n=128):
    """Precompute all nonconstant 4-term APs as index tuples (0-indexed)."""
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _check_validity(colors, aps):
    """Check if coloring satisfies: every nonconstant 4-term AP has >=3 distinct colors."""
    for (a, b, c, d) in aps:
        vals = {colors[a], colors[b], colors[c], colors[d]}
        if len(vals) < 3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_with_backtrack(n, num_colors, aps, deadline, random_seed=42):
    """Greedy with limited backtracking to find valid coloring with num_colors colors."""
    random.seed(random_seed)
    colors = [-1] * n
    
    # For each position, track which APs it's involved in
    pos_to_aps = defaultdict(list)
    for idx, (a, b, c, d) in enumerate(aps):
        for pos in (a, b, c, d):
            pos_to_aps[pos].append(idx)
    
    def is_valid_partial(pos, color):
        """Check if assigning color to pos is consistent with already-assigned positions in its APs."""
        for ap_idx in pos_to_aps[pos]:
            a, b, c, d = aps[ap_idx]
            positions = [a, b, c, d]
            # Count assigned positions and their colors
            assigned_colors = []
            for p in positions:
                if p == pos:
                    assigned_colors.append(color)
                elif colors[p] != -1:
                    assigned_colors.append(colors[p])
                else:
                    # Not yet assigned - can't fully check, but check what we have
                    pass
            
            # If we have 3 or 4 assigned, check if they already violate
            if len(assigned_colors) >= 3:
                distinct = set(assigned_colors)
                if len(distinct) < 3:
                    # Even with remaining unassigned, if we already have <3 distinct
                    # and the remaining can't add new colors... actually they can
                    # So this is only a violation if ALL 4 are assigned and <3 distinct
                    pass
            
            # If all 4 are assigned, check
            all_assigned = all(colors[p] != -1 or p == pos for p in positions)
            if all_assigned:
                full_colors = []
                for p in positions:
                    if p == pos:
                        full_colors.append(color)
                    else:
                        full_colors.append(colors[p])
                if len(set(full_colors)) < 3:
                    return False
        return True
    
    def backtrack(pos, depth):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        # Try colors in order
        color_order = list(range(num_colors))
        random.shuffle(color_order)
        
        for color in color_order:
            if is_valid_partial(pos, color):
                colors[pos] = color
                if backtrack(pos + 1, depth + 1):
                    return True
                colors[pos] = -1
        return False
    
    if backtrack(0, 0):
        return colors
    return None

def _search_periodic(n, num_colors, max_period, deadline):
    """Search for periodic colorings with small periods."""
    aps = _precompute_aps(n)
    best = None
    best_count = 999
    
    for p in range(1, max_period + 1):
        if time.time() > deadline:
            break
        
        # Only try if the number of patterns is manageable
        total_patterns = num_colors ** p
        if total_patterns > 100000:
            continue
        
        for pattern in itertools.product(range(num_colors), repeat=p):
            # Check if pattern uses at least 2 colors (need >=3 in APs)
            if len(set(pattern)) < 2:
                continue
            
            colors = [pattern[i % p] for i in range(n)]
            actual_colors = set(colors)
            if len(actual_colors) > best_count:
                continue
            
            # Validate
            valid = True
            for (a, b, c, d) in aps:
                vals = {colors[a], colors[b], colors[c], colors[d]}
                if len(vals) < 3:
                    valid = False
                    break
            
            if valid:
                cnt = len(actual_colors)
                if cnt < best_count:
                    best_count = cnt
                    best = list(colors)
                if best_count == 2:
                    return best
    
    return best

def _group_theoretic_search(n, deadline):
    """Try group-theoretic coloring using F_2^7 structure."""
    aps = _precompute_aps(n)
    
    # Try various nonlinear functions from F_2^7 to small color sets
    # Position i (0-indexed) maps to 7-bit binary
    # Try: color = f(binary representation of i)
    
    best = None
    best_count = 999
    
    # Try quadratic forms and other low-degree polynomials over F_2
    # For each pair of bits, compute XOR, AND, etc.
    
    # Try: color based on parity of certain bit products
    for mask1 in range(1, 128):
        if time.time() > deadline:
            break
        for mask2 in range(1, 128):
            if time.time() > deadline:
                break
            # color = popcount(i & mask1) XOR popcount(i & mask2) mod 2
            # Or: color = (i & mask1) != 0 XOR (i & mask2) != 0
            colors = []
            for i in range(n):
                b1 = (i & mask1) != 0
                b2 = (i & mask2) != 0
                colors.append(0 if not b1 and not b2 else 1 if b1 and not b2 else 2 if b2 and not b1 else 3)
            
            actual = set(colors)
            if len(actual) > best_count:
                continue
            
            # Validate
            valid = True
            for (a, b, c, d) in aps:
                vals = {colors[a], colors[b], colors[c], colors[d]}
                if len(vals) < 3:
                    valid = False
                    break
            
            if valid:
                cnt = len(actual)
                if cnt < best_count:
                    best_count = cnt
                    best = list(colors)
                if best_count == 2:
                    return best
    
    return best

def construct():
    """Return n=128 integer color labels in 0..127. Every nonconstant four-term progression
    of positions must use AT LEAST three distinct colors. Minimize number of labels used."""
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    n = 128
    aps = _precompute_aps(n)
    
    # First, try to find a 2-coloring (very unlikely but check quickly)
    result = _search_periodic(n, 2, 16, deadline)
    if result is not None:
        if _check_validity(result, aps):
            return result
    
    # Try 3-coloring with periodic patterns
    result = _search_periodic(n, 3, 20, deadline)
    if result is not None:
        if _check_validity(result, aps):
            return result
    
    # Try group-theoretic approach
    if time.time() < deadline:
        result = _group_theoretic_search(n, deadline)
        if result is not None:
            if _check_validity(result, aps):
                return result
    
    # Try greedy with backtracking for 3 colors
    if time.time() < deadline:
        result = _greedy_with_backtrack(n, 3, aps, deadline, random_seed=42)
        if result is not None:
            if _check_validity(result, aps):
                return result
    
    # Try greedy with backtracking for 4 colors
    if time.time() < deadline:
        result = _greedy_with_backtrack(n, 4, aps, deadline, random_seed=42)
        if result is not None:
            if _check_validity(result, aps):
                return result
    
    # Fallback: original greedy approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    
    return colors
# EVOLVE-BLOCK-END
