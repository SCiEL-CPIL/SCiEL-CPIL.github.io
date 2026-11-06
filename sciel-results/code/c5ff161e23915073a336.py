<<<<<<< SEARCH
import time
import random

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _backtrack_3color(n=128, seed=0, time_limit=5.0):
    """Backtracking search for a 3-coloring."""
    rng = random.Random(seed)
    start_time = time.time()
    
    colors = [-1] * n  # -1 means unassigned
    
    # Precompute for each position i, which (a, d) pairs form a 4-AP (a, a+d, a+2d, a+3d) where i is one of the four
    # For forward checking, when we assign position i, we need to check all 4-APs containing i
    # and see if the constraint is satisfied or if it eliminates colors for unassigned positions
    
    # For efficiency, precompute APs
    # APs where position i is the last element (a+3d = i): a = i-3d, d>=1, a>=0
    # APs where position i is the third element (a+2d = i): a = i-2d, d>=1, a>=0, and i+d < n
    # APs where position i is the second element (a+d = i): a = i-d, d>=1, a>=0, and i+2d < n
    # APs where position i is the first element (a = i): d>=1, i+3d < n
    
    def get_aps_containing(i):
        """Yield (a, b, c, e, pos_of_i) for all 4-APs containing position i."""
        ap_list = []
        # i is a+3d: a = i-3d, b = i-2d, c = i-d, e = i
        for d in range(1, (i // 3) + 1):
            a = i - 3 * d
            if a >= 0:
                ap_list.append((a, i - 2 * d, i - d, i, 3))
        # i is a+2d: a = i-2d, b = i-d, c = i, e = i+d
        for d in range(1, ((n - 1 - i) // 1) + 1):
            a = i - 2 * d
            e = i + d
            if a >= 0 and e < n:
                ap_list.append((a, i - d, i, e, 2))
        # i is a+d: a = i-d, b = i, c = i+d, e = i+2d
        for d in range(1, ((n - 1 - i) // 2) + 1):
            a = i - d
            c = i + d
            e = i + 2 * d
            if a >= 0 and e < n:
                ap_list.append((a, i, c, e, 1))
        # i is a: a = i, b = i+d, c = i+2d, e = i+3d
        for d in range(1, ((n - 1 - i) // 3) + 1):
            b = i + d
            c = i + 2 * d
            e = i + 3 * d
            if e < n:
                ap_list.append((i, b, c, e, 0))
        return ap_list
    
    # Cache APs
    ap_cache = [get_aps_containing(i) for i in range(n)]
    
    def is_valid_assignment(i, color):
        """Check if assigning color to position i is consistent with all 4-APs containing i 
        where all other positions are already assigned."""
        for a, b, c, e, pos_idx in ap_cache[i]:
            positions = [a, b, c, e]
            # Check if all other positions are assigned
            other_assigned = True
            for j, p in enumerate(positions):
                if p == i:
                    continue
                if colors[p] == -1:
                    other_assigned = False
                    break
            if other_assigned:
                # All other positions assigned, check constraint
                cols = [color]
                for j, p in enumerate(positions):
                    if p != i:
                        cols.append(colors[p])
                if len(set(cols)) < 3:
                    return False
        return True
    
    def forward_check(i, color):
        """After assigning color to position i, check if any unassigned position 
        in a 4-AP with i now has no valid colors. Returns True if consistent."""
        for a, b, c, e, pos_idx in ap_cache[i]:
            positions = [a, b, c, e]
            unassigned = [p for p in positions if p != i and colors[p] == -1]
            if len(unassigned) == 0:
                # All assigned, verify
                cols = [color] + [colors[p] for p in positions if p != i]
                if len(set(cols)) < 3:
                    return False
            elif len(unassigned) == 1:
                # One unassigned, check if it has at least one valid color
                upos = unassigned[0]
                # The other two positions (besides i and upos) are assigned
                other_two = [p for p in positions if p != i and p != upos]
                c1, c2 = colors[other_two[0]], colors[other_two[1]]
                # For color x for upos: {color, c1, c2, x} must have size >= 3
                # x is in {0,1,2}
                valid_x = []
                for x in range(3):
                    if len({color, c1, c2, x}) >= 3:
                        valid_x.append(x)
                if not valid_x:
                    return False
        return True
    
    def backtrack(pos):
        if time.time() - start_time > time_limit:
            return False
        if pos == n:
            return True
        
        # Try colors in random order
        colors_to_try = rng.sample([0, 1, 2], 3)
        for color in colors_to_try:
            if is_valid_assignment(pos, color):
                colors[pos] = color
                if forward_check(pos, color):
                    if backtrack(pos + 1):
                        return True
                colors[pos] = -1
        return False
    
    if backtrack(0):
        return list(colors)
    return None

def _greedy_3color(n=128, seed=0):
    """Randomized greedy with backtracking fallback."""
    rng = random.Random(seed)
    colors = [None] * n
    
    for i in range(n):
        valid_colors = set([0, 1, 2])
        # Check all 4-APs where i is the last element
        for d in range(1, (i) // 3 + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            if a < 0:
                break
            c1, c2, c3 = colors[a], colors[b], colors[c]
            for color in list(valid_colors):
                if len({c1, c2, c3, color}) < 3:
                    valid_colors.discard(color)
        
        if not valid_colors:
            return None
        
        # Prefer colors that are already used to minimize color count
        used = set(colors[:i])
        preferred = valid_colors & used
        if preferred:
            color = rng.choice(list(preferred))
        else:
            color = rng.choice(list(valid_colors))
        
        colors[i] = color
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    
    # First, get the seed fallback (guaranteed valid)
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # Try to find a 3-coloring
    # First try randomized greedy with many seeds
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        result = _greedy_3color(n, seed)
        if result is not None:
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # Try backtracking with various seeds and time limits
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        num_attempts = min(50, int(remaining_time / 2))
        for seed in range(num_attempts):
            if time.time() > deadline - 2:
                break
            time_limit = min(4.0, deadline - time.time() - 2)
            if time_limit < 0.5:
                break
            result = _backtrack_3color(n, seed, time_limit)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
                        if best_count == 3:
                            return best
    
    return best
=======
import time
import random

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _precompute_aps(n=128):
    """Precompute all 4-APs and which positions they contain."""
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            ap = (a, b, c, e)
            all_aps.append(ap)
            pos_to_aps[a].append(len(all_aps) - 1)
            pos_to_aps[b].append(len(all_aps) - 1)
            pos_to_aps[c].append(len(all_aps) - 1)
            pos_to_aps[e].append(len(all_aps) - 1)
    
    return all_aps, pos_to_aps

def _backtrack_3color(n=128, seed=0, time_limit=5.0):
    """Backtracking search for a 3-coloring with constraint propagation."""
    rng = random.Random(seed)
    start_time = time.time()
    
    all_aps, pos_to_aps = _precompute_aps(n)
    num_aps = len(all_aps)
    
    colors = [-1] * n  # -1 means unassigned
    domains = [set([0, 1, 2]) for _ in range(n)]
    
    def check_ap_satisfied(ap_idx, assigned_color=None, assigned_pos=None):
        """Check if AP at ap_idx is satisfied given current assignments and optional new assignment."""
        a, b, c, e = all_aps[ap_idx]
        positions = [a, b, c, e]
        
        color_set = set()
        for pos in positions:
            if pos == assigned_pos:
                if assigned_color is None:
                    return False  # Cannot check if assigned_pos has no color
                color_set.add(assigned_color)
            else:
                if colors[pos] == -1:
                    return True  # Not fully assigned, can't violate yet
                color_set.add(colors[pos])
        
        return len(color_set) >= 3
    
    def propagate(pos, color):
        """After assigning color to pos, propagate constraints. Returns False if contradiction."""
        for ap_idx in pos_to_aps[pos]:
            a, b, c, e = all_aps[ap_idx]
            positions = [a, b, c, e]
            
            # Count unassigned positions in this AP
            unassigned = []
            assigned_colors = []
            for p in positions:
                if p == pos:
                    assigned_colors.append(color)
                elif colors[p] == -1:
                    unassigned.append(p)
                else:
                    assigned_colors.append(colors[p])
            
            if len(unassigned) == 0:
                # All assigned, check constraint
                if len(set(assigned_colors)) < 3:
                    return False
            elif len(unassigned) == 1:
                # One unassigned, check if it has any valid color
                upos = unassigned[0]
                valid_colors_for_upos = set()
                for x in domains[upos]:
                    if len(set(assigned_colors + [x])) >= 3:
                        valid_colors_for_upos.add(x)
                if not valid_colors_for_upos:
                    return False
                # Reduce domain
                domains[upos] = valid_colors_for_upos
            elif len(unassigned) == 2:
                # Two unassigned, check if there exist colors for both
                u1, u2 = unassigned
                # For each color for u1, check if u2 can have a valid color
                found_valid = False
                for x1 in domains[u1]:
                    for x2 in domains[u2]:
                        if len(set(assigned_colors + [x1, x2])) >= 3:
                            found_valid = True
                            break
                    if found_valid:
                        break
                if not found_valid:
                    return False
        return True
    
    def backtrack():
        if time.time() - start_time > time_limit:
            return False
        
        # Find unassigned position with smallest domain (MRV heuristic)
        min_domain_size = 4
        min_pos = -1
        for i in range(n):
            if colors[i] == -1 and len(domains[i]) < min_domain_size:
                min_domain_size = len(domains[i])
                min_pos = i
        
        if min_pos == -1:
            return True  # All assigned
        
        # Try colors in random order, but prefer colors that lead to more constraints
        colors_to_try = list(domains[min_pos])
        rng.shuffle(colors_to_try)
        
        for color in colors_to_try:
            if time.time() - start_time > time_limit:
                return False
            
            # Save state
            old_color = colors[min_pos]
            old_domain = domains[min_pos].copy()
            old_domains = [d.copy() for d in domains]
            
            colors[min_pos] = color
            domains[min_pos] = {color}
            
            if propagate(min_pos, color):
                if backtrack():
                    return True
            
            # Restore state
            colors[min_pos] = old_color
            domains[min_pos] = old_domain
            domains = old_domains
        
        return False
    
    if backtrack():
        return list(colors)
    return None

def _greedy_3color(n=128, seed=0):
    """Randomized greedy with backtracking fallback."""
    rng = random.Random(seed)
    colors = [None] * n
    
    for i in range(n):
        valid_colors = set([0, 1, 2])
        # Check all 4-APs where i is the last element
        for d in range(1, (i) // 3 + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            if a < 0:
                break
            c1, c2, c3 = colors[a], colors[b], colors[c]
            for color in list(valid_colors):
                if len({c1, c2, c3, color}) < 3:
                    valid_colors.discard(color)
        
        if not valid_colors:
            return None
        
        # Prefer colors that are already used to minimize color count
        used = set(colors[:i])
        preferred = valid_colors & used
        if preferred:
            color = rng.choice(list(preferred))
        else:
            color = rng.choice(list(valid_colors))
        
        colors[i] = color
    
    return colors

def _local_search_3color(n=128, seed=0, time_limit=5.0):
    """Local search to improve a 3-coloring."""
    rng = random.Random(seed)
    start_time = time.time()
    
    all_aps, pos_to_aps = _precompute_aps(n)
    
    # Start with a random 3-coloring
    colors = [rng.choice([0, 1, 2]) for _ in range(n)]
    
    def count_violations():
        count = 0
        for a, b, c, e in all_aps:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                count += 1
        return count
    
    best_colors = colors.copy()
    best_violations = count_violations()
    
    while time.time() - start_time < time_limit:
        # Pick a random position and try changing its color
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        new_color = rng.choice([0, 1, 2])
        
        if new_color == old_color:
            continue
        
        colors[pos] = new_color
        violations = count_violations()
        
        if violations < best_violations:
            best_violations = violations
            best_colors = colors.copy()
            if best_violations == 0:
                return best_colors
        else:
            colors[pos] = old_color
    
    return best_colors if best_violations == 0 else None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    
    # First, get the seed fallback (guaranteed valid)
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # Try to find a 3-coloring
    # First try randomized greedy with many seeds
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        result = _greedy_3color(n, seed)
        if result is not None:
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # Try backtracking with various seeds and time limits
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        num_attempts = min(50, int(remaining_time / 2))
        for seed in range(num_attempts):
            if time.time() > deadline - 2:
                break
            time_limit = min(4.0, deadline - time.time() - 2)
            if time_limit < 0.5:
                break
            result = _backtrack_3color(n, seed, time_limit)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
                        if best_count == 3:
                            return best
    
    # Try local search as a final attempt
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        for seed in range(20):
            if time.time() > deadline - 2:
                break
            time_limit = min(3.0, deadline - time.time() - 2)
            if time_limit < 0.5:
                break
            result = _local_search_3color(n, seed, time_limit)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
                        if best_count == 3:
                            return best
    
    return best
>>>>>>> REPLACE
