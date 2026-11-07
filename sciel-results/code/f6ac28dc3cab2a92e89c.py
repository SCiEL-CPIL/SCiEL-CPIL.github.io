import time
import random
from math import isqrt

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

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
    # Optimization: check if p*x is in prods for any p in prods
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

def _local_search(selected, prods, deadline, max_rounds=50):
    """Local search: try adding, 1-for-2, or 2-for-3 exchanges to improve set size."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # 1. Try adding any candidate (prefer larger values first)
        for x in sorted(candidates, key=lambda v: -v):
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
            # 2. Try 1-for-2 swaps: remove one element, try to add two
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                remaining = selected[:i] + selected[i+1:]
                rem_prods = _subset_products_set(remaining)
                
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
            # 3. Try 2-for-3 swaps: remove two elements, try to add three
            if len(selected) >= 2:
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, len(selected)):
                        if time.time() > deadline:
                            break
                        rem_sel = [selected[m] for m in range(len(selected)) if m != i and m != j]
                        rem_prods = _subset_products_set(rem_sel)
                        
                        added_any = False
                        for a in range(len(candidates)):
                            if time.time() > deadline:
                                break
                            x1 = candidates[a]
                            ok1, sel1, prods1 = _try_add(rem_sel, rem_prods, x1)
                            if not ok1:
                                continue
                            for b in range(a+1, len(candidates)):
                                if time.time() > deadline:
                                    break
                                x2 = candidates[b]
                                ok2, sel2, prods2 = _try_add(sel1, prods1, x2)
                                if not ok2:
                                    continue
                                for c in range(b+1, len(candidates)):
                                    if time.time() > deadline:
                                        break
                                    x3 = candidates[c]
                                    ok3, sel3, prods3 = _try_add(sel2, prods2, x3)
                                    if ok3 and len(sel3) > len(selected):
                                        selected = sel3
                                        prods = prods3
                                        improved = True
                                        added_any = True
                                        break
                                if added_any:
                                    break
                            if added_any:
                                break
                        if added_any:
                            break
                    if improved:
                        break
                if improved:
                    continue
            
            break
            
    return selected, prods

def construct():
    if time.time() > _DEADLINE:
        return [2]

    # Start with the strong heuristic set
    selected = _construct_initial_set()
    prods = _subset_products_set(selected)
    best_selected = list(selected)
    best_size = len(selected)
    
    # Refine with local search
    if len(selected) < 18 and time.time() < _DEADLINE:
        selected, prods = _local_search(selected, prods, _DEADLINE)
        if len(selected) > best_size:
            best_selected = list(selected)
            best_size = len(selected)
    
    # Try alternative greedy constructions
    if time.time() < _DEADLINE and best_size < 18:
        for order in [list(range(40, 1, -1)), 
                      [p for p in range(2, 41) if all(p % d != 0 for d in range(2, int(p**0.5)+1))],
                      [c for c in range(2, 41) if any(c % d == 0 for d in range(2, int(c**0.5)+1))]]:
            if time.time() > _DEADLINE:
                break
            gsel, gprods = [], {1}
            for x in order:
                if time.time() > _DEADLINE:
                    break
                ok, gsel, gprods = _try_add(gsel, gprods, x)
            if len(gsel) > best_size:
                best_selected = list(gsel)
                best_size = len(gsel)
    
    # Final greedy fill on the best set found
    if time.time() < _DEADLINE and best_size < 18:
        selected = list(best_selected)
        prods = _subset_products_set(selected)
        used = set(selected)
        for x in range(2, 41):
            if x in used:
                continue
            if len(selected) >= 18:
                break
            ok, selected, prods = _try_add(selected, prods, x)
            if ok:
                used.add(x)
        if len(selected) > best_size:
            best_selected = list(selected)
            best_size = len(selected)
    
    selected = best_selected

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
