"""Rewritten program: backtracking search over integer Sidon sets with 0,
covering differences 1..30, minimizing the maximum element."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill
_TARGET_MASK = (1 << 30) - 1

# ---------------------------------------------------------------------------
# Fast helpers
# ---------------------------------------------------------------------------

def _is_sidon(s):
    """True iff all positive differences are distinct."""
    n = len(s)
    seen = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d in seen:
                return False
            seen.add(d)
    return True

def _covers_1_to_30(s):
    d = set()
    for i in range(len(s)):
        si = s[i]
        for j in range(i + 1, len(s)):
            d.add(s[j] - si)
    for x in range(1, 31):
        if x not in d:
            return False
    return True

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

# ---------------------------------------------------------------------------
# Greedy baseline (fast, deterministic)
# ---------------------------------------------------------------------------

def _greedy_construct():
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        found = False
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            dl = diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    return a

# ---------------------------------------------------------------------------
# Backtracking search
# ---------------------------------------------------------------------------

def _backtracking_search(D, time_limit):
    """
    Search for a valid Sidon set in [0, D] covering 1..30.
    Returns the set if found, None otherwise.
    """
    best_sol = None
    nodes = [0]
    
    def dfs(current_set, covered_mask, used_diffs):
        nonlocal best_sol
        if time.time() > time_limit:
            return False
        
        nodes[0] += 1
        
        if covered_mask == _TARGET_MASK:
            best_sol = current_set[:]
            return True
            
        if len(current_set) >= 32:
            return False
            
        last_val = current_set[-1]
        
        # Pruning: if we can't cover remaining diffs with remaining slots
        remaining_slots = 32 - len(current_set)
        uncovered = _TARGET_MASK & ~covered_mask
        # Each new element can cover at most (len(current_set)+1) new diffs in 1..30
        # But more tightly: each new element x adds diffs x-y for y in current_set
        # So at most len(current_set) + 1 diffs per element (including to future elements)
        # Simple bound: need at least ceil(popcount(uncovered) / (len(current_set)+1)) more elements
        popcount_uncovered = bin(uncovered).count("1")
        max_new_per_elem = len(current_set) + 1
        if popcount_uncovered > remaining_slots * max_new_per_elem:
            return False
        
        # Generate candidates x > last_val
        candidates = []
        for x in range(last_val + 1, D + 1):
            valid = True
            new_diffs = []
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
                    
            improvement = bin(new_mask ^ covered_mask).count("1")
            if improvement == 0:
                continue
                
            candidates.append((improvement, x, new_diffs, new_mask))
            
        if not candidates:
            return False
            
        # Sort: prefer higher improvement, then smaller x (to keep diameter small)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        
        for imp, x, new_diffs, new_mask in candidates:
            new_set = current_set + [x]
            new_used = used_diffs.union(new_diffs)
            
            if dfs(new_set, new_mask, new_used):
                return True
                
        return False

    if dfs([0], 0, set()):
        return best_sol
    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    
    # Get a baseline solution to ensure we have something valid
    baseline = _greedy_construct()
    best_val = max(baseline)
    best_set = baseline
    
    # Try to find a smaller diameter using backtracking
    # Start from a lower bound.
    # We need at least 9 elements.
    # A safe lower bound for diameter is 50.
    # We iterate D upwards.
    
    # To save time, we can skip D values if we are close to deadline
    # But backtracking is fast for small D.
    
    # Start from a lower bound.
    # A Sidon set covering 1..30 needs at least ~9 elements.
    # With k elements, we have k*(k-1)/2 differences.
    # Need at least 30 differences, so k*(k-1)/2 >= 30 => k >= 9.
    # The minimum diameter for a (v,k,lambda) difference set is related to k.
    # For small sets, diameter is at least ~30-40.
    
    start_D = 40
    end_D = best_val
    
    # Adaptive time per D: give more time to smaller D values
    for D in range(start_D, end_D + 1):
        if time.time() > _DEADLINE - 3:
            break
            
        # Smaller D gets more time (harder to find if it exists)
        # Larger D gets less time (easier if it exists)
        if D < 60:
            per_d_time = 8.0
        elif D < 80:
            per_d_time = 5.0
        elif D < 100:
            per_d_time = 3.0
        else:
            per_d_time = 2.0
        
        t_start = time.time()
        limit = t_start + per_d_time
        
        sol = _backtracking_search(D, limit)
        
        if sol:
            # This is the minimum diameter found so far
            return sol
            
    # If no better solution found, return baseline
    return best_set

# EVOLVE-BLOCK-END
