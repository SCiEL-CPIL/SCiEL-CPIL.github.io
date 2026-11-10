"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

_DEADLINE_SECONDS = 85  # safety margin before 90s kill

# Precompute GCD table for 1..128
_MAX_VAL = 128
_gcd_table = [[0] * (_MAX_VAL + 1) for _ in range(_MAX_VAL + 1)]
for _i in range(1, _MAX_VAL + 1):
    for _j in range(_i, _MAX_VAL + 1):
        _g = gcd(_i, _j)
        _gcd_table[_i][_j] = _g
        _gcd_table[_j][_i] = _g

def _can_add(x, selected):
    """Check if adding x to selected keeps the set valid (no bad triple formed)."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        g_ax = _gcd_table[a][x]
        for j in range(i + 1, n):
            b = selected[j]
            g_ab = _gcd_table[a][b]
            if g_ab == g_ax and _gcd_table[b][x] == g_ab:
                return False
    return True

def _can_add_fast(x, selected, gcd_cache):
    """Optimized can_add using precomputed gcd values for x against all selected."""
    n = len(selected)
    # Precompute gcd(x, selected[i]) for all i
    gx = [0] * n
    for i in range(n):
        gx[i] = _gcd_table[selected[i]][x]
    for i in range(n):
        a = selected[i]
        g_ax = gx[i]
        for j in range(i + 1, n):
            b = selected[j]
            g_ab = _gcd_table[a][b]
            if g_ab == g_ax and gx[j] == g_ab:
                return False
    return True

def _greedy_pass(order):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    for x in order:
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_improve_add(selected, candidates, deadline):
    """Try to add back excluded candidates in random order, multiple passes."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    
    # Multiple passes to find more additions
    max_passes = 10
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        if not to_try:
            break
        random.shuffle(to_try)
        added_any = False
        for x in to_try:
            if time.time() > deadline:
                break
            if _can_add(x, improved):
                improved.append(x)
                in_set.add(x)
                to_try.remove(x)
                added_any = True
        if not added_any:
            break
    return improved

def _local_improve_swap(selected, candidates, deadline, max_swaps=100):
    """Try swap operations: remove one element, then try to add multiple others.
    Also try 2-for-1 swaps (remove 2, add 3 or more)."""
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        # Randomly decide between 1-for-k and 2-for-k swaps
        use_two = len(improved) >= 2 and random.random() < 0.3
        
        if use_two:
            # Remove 2 elements
            idx1 = random.randint(0, len(improved) - 1)
            idx2 = random.randint(0, len(improved) - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(improved) - 1)
            removed1 = improved[idx1]
            removed2 = improved[idx2]
            # Remove in reverse order to preserve indices
            if idx1 > idx2:
                improved.pop(idx1)
                improved.pop(idx2)
            else:
                improved.pop(idx2)
                improved.pop(idx1)
            in_set.discard(removed1)
            in_set.discard(removed2)
            
            # Try to add multiple candidates
            added_any = False
            remaining = [e for e in excluded if e not in in_set]
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline:
                    break
                if _can_add(x, improved):
                    improved.append(x)
                    in_set.add(x)
                    added_any = True
            
            if not added_any:
                # Put them back
                improved.extend([removed1, removed2])
                in_set.add(removed1)
                in_set.add(removed2)
            else:
                excluded = [c for c in candidates if c not in in_set]
        else:
            # Pick a random element to remove
            idx = random.randint(0, len(improved) - 1)
            removed = improved[idx]
            improved.pop(idx)
            in_set.discard(removed)
            
            # Try to add multiple candidates
            added_any = False
            remaining = [e for e in excluded if e not in in_set]
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline:
                    break
                if _can_add(x, improved):
                    improved.append(x)
                    in_set.add(x)
                    added_any = True
            
            if not added_any:
                # Put it back
                improved.append(removed)
                in_set.add(removed)
            else:
                # Update excluded list
                excluded = [c for c in candidates if c not in in_set]
    
    return improved

def _verify_validity(selected):
    """Verify that the selected set has no bad triples."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i + 1, n):
            b = selected[j]
            g_ab = _gcd_table[a][b]
            for k in range(j + 1, n):
                c = selected[k]
                if _gcd_table[a][c] == g_ab and _gcd_table[b][c] == g_ab:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, 129))
    
    # Helper: count total prime factors (with multiplicity) for ordering heuristic
    def factor_count(n):
        cnt = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                cnt += 1
                n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt
    
    fc = {x: factor_count(x) for x in all_nums}
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Sequential greedy (original)
    if time.time() < deadline:
        s = _greedy_pass(all_nums)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 2: Prime-factor-rich first greedy
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (-fc[x], x))
        s = _greedy_pass(order)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 3: Reverse order greedy
    if time.time() < deadline:
        s = _greedy_pass(list(reversed(all_nums)))
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 4: Many random shuffle greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 500
    for _ in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            # Try local improvement on the new best
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved = _local_improve_swap(best_set, all_nums, deadline, max_swaps=50)
                    if len(improved) > best_len:
                        best_set, best_len = improved, len(improved)
    
    # Strategy 5: More aggressive local search with multiple restarts
    if time.time() < deadline and best_set:
        # Try multiple rounds of swap-based local search
        for _round in range(5):
            if time.time() > deadline:
                break
            improved = _local_improve_swap(best_set, all_nums, deadline, max_swaps=200)
            if len(improved) > best_len:
                best_set, best_len = improved, len(improved)
                improved = _local_improve_add(best_set, all_nums, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
            else:
                break
    
    # Final local improvement on best set
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
    if time.time() < deadline:
        improved = _local_improve_swap(best_set, all_nums, deadline, max_swaps=200)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
    
    # Ensure nonempty and valid
    if not best_set:
        best_set = [1]
    elif not _verify_validity(best_set):
        # Fallback: rebuild a valid set from scratch
        best_set = _greedy_pass(all_nums)
        if not best_set:
            best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
