import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Evaluator kills at 90s; keep a 5s safety margin

_NUMS = list(range(2, 41))
_N = len(_NUMS)

# Pre-compute prime factorization vectors for quick conflict pre-filtering (optional, but exact check is robust)
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PN = len(_PRIMES)
_EXP = {}

def _build_exp_table():
    for n in range(2, 41):
        m = n
        vec = [0] * _PN
        for i, p in enumerate(_PRIMES):
            while m % p == 0:
                vec[i] += 1
                m //= p
        _EXP[n] = tuple(vec)

_build_exp_table()

def _subset_products_set(nums):
    """Compute the set of all subset products of nums."""
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """Check if x can be added to the set with current subset products 'prods'."""
    # If x * p is already in prods for some p in prods, then there is a collision.
    # Specifically, if x * p1 == p2, then {x} * subset(p1) == subset(p2).
    # We need to check if x * p is in prods for ANY p in prods.
    for p in prods:
        if (p * x) in prods:
            return False
    return True

def _try_add(selected, prods, x):
    """Attempt to add x. Returns (success, new_selected, new_prods)."""
    if not _can_add(prods, x):
        return False, selected, prods
    
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    
    return True, selected + [x], new_prods

def _local_search(selected, prods, deadline, max_rounds=50):
    """
    Perform local search to maximize the size of the set.
    Strategies:
    1. Add any compatible element.
    2. Swap: Remove one, add one different.
    3. Net +1: Remove one, add two different.
    """
    selected = list(selected)
    prods = _subset_products_set(selected)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
            
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # Strategy 1: Try adding any candidate
        # Sort candidates by "safety" (fewer conflicts) to find a valid add faster?
        # Or just try all.
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sel, new_prods = _try_add(selected, prods, x)
            if ok:
                selected = new_sel
                prods = new_prods
                improved = True
                used = set(selected)
                candidates = [c for c in range(2, 41) if c not in used]
                break # Restart loop with new state
        
        if not improved and len(selected) > 0:
            # Strategy 2: Single Swap (Remove i, Add j)
            # We want to maintain size, but maybe position for a future add?
            # Actually, if we can't add, swapping doesn't increase size.
            # But it might unblock a future add.
            # However, the prompt says "Maximize number". 
            # If we are stuck at 17, we need to get to 18.
            # Swapping 17 -> 17 doesn't help directly unless it allows an 18th add.
            # So we should look for swaps that allow an add.
            # Or just try to find a configuration of 18 directly.
            
            # Let's try "Remove 1, Add 2" (Net +1)
            # This is expensive: O(N * M^2)
            # N ~ 17, M ~ 22. 17 * 484 ~ 8000 checks.
            # Each check is O(2^16) ~ 65k ops. Total ~ 500M ops. Might be too slow in Python.
            
            # Optimization: 
            # If we remove s_i, we get prods_without_i.
            # We need to find x, y in candidates such that x, y can be added to prods_without_i.
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_prods = _subset_products_set(remaining)
                
                # Try to add two elements from candidates
                # We can iterate x, then y.
                # To speed up, we can pre-filter candidates that are compatible with rem_prods.
                valid_adds = []
                for x in candidates:
                    if x == removed:
                        continue
                    if _can_add(rem_prods, x):
                        valid_adds.append(x)
                
                # Now find a pair (x, y) from valid_adds that can be added together
                # We need to check if x and y can be added to rem_prods simultaneously.
                # This is equivalent to: can we add x to rem_prods, and then y to the result?
                # Or check if subset sums of {x, y} intersect with rem_prods?
                
                for idx_x, x in enumerate(valid_adds):
                    if time.time() > deadline:
                        break
                    # Compute prods after adding x
                    prods_with_x = set(rem_prods)
                    for p in rem_prods:
                        prods_with_x.add(p * x)
                    
                    # Now try to add y
                    for y in valid_adds[idx_x+1:]:
                        if time.time() > deadline:
                            break
                        if _can_add(prods_with_x, y):
                            # Success!
                            selected = remaining + [x, y]
                            prods = _subset_products_set(selected)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # If we can't add, maybe we should try random restarts in the outer loop
                # or accept this local optimum.
                break
                
    return selected, prods

def construct():
    if time.time() > _DEADLINE:
        return [2]
    
    best_selected = None
    best_len = 0
    
    # Phase 1: Deterministic Heuristic
    # Start with primes, then add composites
    initial_candidates = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # Add squares
    initial_candidates += [4, 9, 25]
    # Add other composites
    rest = [x for x in range(2, 41) if x not in initial_candidates]
    initial_candidates += rest
    
    selected, prods = [], {1}
    for x in initial_candidates:
        if len(selected) >= 18:
            break
        ok, selected, prods = _try_add(selected, prods, x)
        if ok:
            pass # already updated
    
    if len(selected) > best_len:
        best_len = len(selected)
        best_selected = list(selected)
    
    if best_len < 18 and time.time() < _DEADLINE:
        selected, prods = _local_search(selected, prods, _DEADLINE)
        if len(selected) > best_len:
            best_len = len(selected)
            best_selected = list(selected)
            
    # Phase 2: Random Restarts
    rng = random.Random(42)
    # Use remaining time for random restarts
    remaining_time = _DEADLINE - time.time()
    # Estimate time per restart. Local search is heavy.
    # Let's do a few fast greedy + light local search.
    
    for restart in range(100): # Cap iterations, time check inside
        if time.time() > _DEADLINE - 2.0:
            break
        
        # Random order
        candidates = list(range(2, 41))
        rng.shuffle(candidates)
        
        selected, prods = [], {1}
        for x in candidates:
            if len(selected) >= 18:
                break
            ok, selected, prods = _try_add(selected, prods, x)
            if ok:
                pass
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_selected = list(selected)
            if best_len >= 18:
                break
                
        # Light local search: only try "Add" and "Remove 1 Add 1" (swap)
        # Full "Remove 1 Add 2" is too slow for many restarts
        if len(selected) < 18 and time.time() < _DEADLINE - 5.0:
             # Try to add
             used = set(selected)
             cands = [c for c in range(2, 41) if c not in used]
             for x in cands:
                 if time.time() > _DEADLINE - 5.0: break
                 ok, new_sel, new_prods = _try_add(selected, prods, x)
                 if ok:
                     selected = new_sel
                     prods = new_prods
                     break
             
             if len(selected) > best_len:
                 best_len = len(selected)
                 best_selected = list(selected)
                 if best_len >= 18:
                     break
                 
             # Try swaps to unblock
             # This is a simplified local search
             for _ in range(5): # 5 rounds of swap
                 if time.time() > _DEADLINE - 5.0: break
                 improved = False
                 used = set(selected)
                 cands = [c for c in range(2, 41) if c not in used]
                 for i in range(len(selected)):
                     if time.time() > _DEADLINE - 5.0: break
                     rem = selected[i]
                     rem_sel = selected[:i] + selected[i+1:]
                     rem_prods = _subset_products_set(rem_sel)
                     for x in cands:
                         if x == rem: continue
                         ok, new_sel, new_prods = _try_add(rem_sel, rem_prods, x)
                         if ok:
                             # Check if this swap allows an add?
                             # Just take the swap if it doesn't reduce size (it keeps size)
                             # But we want to increase size.
                             # So we only take swap if it leads to a state where we CAN add?
                             # That's complex. 
                             # Let's just take the swap if it results in a "better" set? 
                             # Hard to define better.
                             # Let's skip swaps in random restarts to save time, 
                             # and rely on the initial greedy + the first heavy local search.
                             pass
                             
    # Final verification
    if best_selected is None:
        best_selected = [2]
        
    final_prods = _subset_products_set(best_selected)
    if len(final_prods) != 2 ** len(best_selected):
        # Fallback to a known valid set
        best_selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Type check
    out = []
    seen = set()
    for x in best_selected:
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            out.append(int(x))
            
    if not out:
        out = [2]
        
    return out
