import time
import random
from math import isqrt

_START_TIME = time.time()
_DEADLINE = _START_TIME + 78.0

def _subset_products_set(nums):
    """Return the set of all subset products of nums (including empty product 1)."""
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            q = p * x
            add.add(q)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct."""
    # Check if p*x is in prods for any p in prods.
    # Equivalently: exists q in prods such that q % x == 0 and q//x in prods.
    # We iterate over prods and check both directions to minimize multiplications.
    for p in prods:
        if (p * x) in prods:
            return False
    return True

def _try_add(selected, prods, x):
    """Try to add x to selected; returns (success, new_selected, new_prods)."""
    if not _can_add(prods, x):
        return False, selected, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    new_selected = selected + [x]
    return True, new_selected, new_prods

def _construct_initial_set():
    """
    Construct a high-quality initial set using the 'pure powers' heuristic.
    Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    Squares: 4 (2^2), 9 (3^2), 25 (5^2)
    Fourth power: 16 (2^4)
    This set of 16 is known to be valid.
    """
    return [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]

def _local_search(selected, prods, deadline, max_rounds=20):
    """Local search: try swapping elements or adding elements to improve the set size."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # 1. Try adding any candidate
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
                if len(selected) >= 18:
                    break
        
        if not improved:
            # 2. Try single swaps: remove one, add one different element
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_prods = _subset_products_set(remaining)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x)
                    if ok1:
                        # Check if this is actually an improvement (different element)
                        if x != removed:
                            selected = sel1
                            prods = prods1
                            improved = True
                            break
                if improved:
                    break
            
        if not improved:
            # 3. Try swapping: remove one, add two
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                # Remove selected[i]
                remaining = selected[:i] + selected[i+1:]
                rem_prods = _subset_products_set(remaining)
                
                # Try adding two candidates
                added_any = False
                for j in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    x1 = candidates[j]
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x1)
                    if not ok1:
                        continue
                    for k in range(j+1, len(candidates)):
                        if time.time() > deadline:
                            break
                        x2 = candidates[k]
                        ok2, sel2, prods2 = _try_add(sel1, prods1, x2)
                        if ok2 and len(sel2) > len(selected):
                            selected = sel2
                            prods = prods2
                            improved = True
                            added_any = True
                            break
                    if added_any:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return selected, prods

def construct():
    if time.time() > _DEADLINE:
        return [2]

    # Start with the strong heuristic set
    selected = _construct_initial_set()
    prods = _subset_products_set(selected)
    
    # Refine with local search
    if len(selected) < 18 and time.time() < _DEADLINE:
        selected, prods = _local_search(selected, prods, _DEADLINE)

    # Final greedy fill
    if time.time() < _DEADLINE and len(selected) < 18:
        used = set(selected)
        for x in range(2, 41):
            if x in used:
                continue
            if len(selected) >= 18:
                break
            ok, selected, prods = _try_add(selected, prods, x)
            if ok:
                used.add(x)

    # Ensure validity and type correctness
    # Re-verify the final set
    final_prods = _subset_products_set(selected)
    if len(final_prods) != 2 ** len(selected):
        # Fallback to a known valid set if something went wrong
        selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Deduplicate and filter
    seen = set()
    out = []
    for x in selected:
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            out.append(int(x))
            
    if not out:
        out = [2]
        
    return out
