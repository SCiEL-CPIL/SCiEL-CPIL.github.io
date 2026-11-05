import time
import random
import itertools

_DEADLINE = 85.0

def _zero_sum_size_ge_2(zs):
    """
    Check if 0 can be formed as a sum of a subset of zs with size >= 2.
    zs are integers (can be negative, zero).
    Uses DP with bitsets.
    """
    n = len(zs)
    if n < 2:
        return False
    
    pos = [z for z in zs if z > 0]
    neg = [-z for z in zs if z < 0]
    zeros = sum(1 for z in zs if z == 0)
    
    if zeros >= 2:
        return True
    
    total_pos = sum(pos)
    total_neg = sum(neg)
    
    offset = total_neg
    max_bit = total_pos + total_neg + 1
    
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    reachable = [True] + [False] * n
    
    for z in zs:
        if z == 0:
            continue
        for k in range(n, 0, -1):
            if not reachable[k-1]:
                continue
            if z > 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            dp[k] |= shifted
            reachable[k] = True
    
    for k in range(2, n + 1):
        if reachable[k] and (dp[k] >> offset) & 1:
            return True
    return False

def _can_form_sum(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals with size >= 1.
    """
    if not vals:
        return False
        
    n = len(vals)
    max_abs = sum(abs(v) for v in vals)
    if abs(target) > max_abs:
        return False
        
    offset = max_abs
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for v in vals:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= shifted
            
    bit_idx = offset + target
    for k in range(1, n + 1):
        if (dp[k] >> bit_idx) & 1:
            return True
    return False

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    if len(others) < 2:
        return False
    
    zs = [y - a for y in others]
    return _zero_sum_size_ge_2(zs)

def _is_valid_incremental(selected, x):
    """
    Check if x can be added to selected without violating constraints.
    """
    if len(selected) >= 2:
        if _zero_sum_size_ge_2([y - x for y in selected]):
            return False

    for y in selected:
        target = y - x
        if target == 0:
            continue
        vals = [z - y for z in selected if z != y]
        if not vals:
            continue
        if _can_form_sum(target, vals):
            return False
            
    return True

def _precompute_conflict_graph(n_max=512):
    """
    Precompute which pairs (a, b) form conflicts with a third element.
    For each x in [1, n_max], find all pairs (i, j) such that x is the mean of {i, j}.
    Also find hyperedges: subsets S of size >= 2 whose mean is an integer.
    We'll build a conflict matrix: conflict[x][y] = True if adding both creates a violation.
    Actually, we need: for each candidate x, which other candidates y would make x
    the mean of a subset containing y.
    
    Simpler approach: for each x, compute the set of "forbidden partners" - 
    elements y such that {x, y} plus some subset of others could make x a mean.
    
    For greedy, we'll track: for each candidate x, the number of already-selected
    elements that conflict with x (i.e., x would be mean of a subset of selected, 
    or some selected element would be mean of a subset including x).
    """
    # For each x, precompute: which subsets of [1,512]\{x} have mean exactly x?
    # That's expensive. Instead, for incremental checking, we'll just use the DP.
    # But for the greedy weight, we can precompute pairwise conflicts:
    # x and y conflict if 2x = x + y (impossible) or if there exist z, w such that
    # the set {x, y, z, w} has x as mean of {y, z, w} or similar.
    
    # Simpler: precompute for each x, the set of y such that x is the mean of {y} 
    # (impossible since need size >= 2) or mean of {y, z} for some z.
    # x is mean of {y, z} iff 2x = y + z iff z = 2x - y.
    # So for each x, the "pair partners" are: for each y != x, if 2x - y is in [1,512] 
    # and != x, then {y, 2x-y} is a pair whose mean is x.
    
    # For the greedy heuristic, define conflict_weight(x, selected) as:
    # 1. Number of pairs (a, b) in selected such that x = (a+b)/2
    # 2. For each y in selected, number of pairs (a, b) in selected\{y} such that y = (a+b+x)/3 or y is mean of some subset including x
    # This is complex. Let's use a simpler proxy:
    # weight(x) = number of selected elements y such that 2x - y is also in selected (pair conflict)
    #           + number of selected elements y such that 3x - 2y is in selected (triple conflict approximation)
    
    pass

def _greedy_conflict_weight(x, selected, sel_set):
    """
    Compute the conflict weight of adding x to selected.
    Lower weight = better candidate.
    """
    weight = 0
    n_sel = len(selected)
    
    # Count pairs (a, b) in selected with a + b = 2x
    for i in range(n_sel):
        for j in range(i+1, n_sel):
            if selected[i] + selected[j] == 2 * x:
                weight += 10  # high penalty for direct pair conflict
    
    # Count for each y in selected: is y the mean of a subset including x?
    # Approximate: check if 2y - x is in selected (y is mean of {x, 2y-x})
    for y in selected:
        partner = 2 * y - x
        if partner in sel_set and partner != y and partner != x:
            weight += 5
    
    # Approximate triple: y is mean of {x, a, b} where a + b = 3y - x
    # For each y, check if there exist a, b in selected\{y} with a + b = 3y - x
    for y in selected:
        target_sum = 3 * y - x
        found = False
        for i in range(n_sel):
            if selected[i] == y:
                continue
            for j in range(i+1, n_sel):
                if selected[j] == y:
                    continue
                if selected[i] + selected[j] == target_sum:
                    found = True
                    break
            if found:
                break
        if found:
            weight += 3
    
    return weight

def construct():
    start_time = time.time()
    best_set = []
    candidates = list(range(1, 513))
    
    # Multiple strategies:
    # 1. Conflict-weight greedy with different initial seeds
    # 2. Spread order
    # 3. Random orders
    
    strategies = []
    
    # Strategy 1: Conflict-weight greedy, trying different first elements
    for first_seed in [1, 256, 512, 128, 384, 64, 448]:
        strategies.append(('greedy_seed', first_seed))
    
    # Strategy 2: Spread orders
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(hi)
        if lo != hi:
            spread.append(lo)
        lo += 1
        hi -= 1
    strategies.append(('order', spread))
    strategies.append(('order', spread[::-1]))
    
    # Strategy 3: Random orders
    rng = random.Random(42)
    for _ in range(20):
        r = candidates[:]
        rng.shuffle(r)
        strategies.append(('order', r))
    
    for strat_type, data in strategies:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        sel_set = set()
        
        if strat_type == 'greedy_seed':
            # Start with the seed
            seed_val = data
            selected.append(seed_val)
            sel_set.add(seed_val)
            
            # Remaining candidates
            remaining = [x for x in candidates if x != seed_val]
            
            while len(selected) < 48:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                
                # Find candidate with minimum conflict weight
                best_x = None
                best_weight = float('inf')
                
                for x in remaining:
                    w = _greedy_conflict_weight(x, selected, sel_set)
                    if w < best_weight:
                        best_weight = w
                        best_x = x
                    # Early termination if weight is 0
                    if w == 0:
                        break
                
                if best_x is None:
                    break
                
                # Verify with full check
                if _is_valid_incremental(selected, best_x):
                    selected.append(best_x)
                    sel_set.add(best_x)
                    remaining.remove(best_x)
                else:
                    # Remove from remaining to avoid re-checking
                    remaining.remove(best_x)
        
        elif strat_type == 'order':
            for x in data:
                if time.time() - start_time > _DEADLINE - 5:
                    break
                if len(selected) >= 48:
                    break
                if x in sel_set:
                    continue
                if _is_valid_incremental(selected, x):
                    selected.append(x)
                    sel_set.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Final verification
    if best_set:
        C = best_set[:]
        valid = True
        for i in range(len(C)):
            if _is_mean_of_subset(C, i):
                valid = False
                break
        if valid:
            return best_set
        else:
            # Fallback: return a known valid small set
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
