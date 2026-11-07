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
    """
    return [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]

def _local_search(selected, prods, deadline, max_rounds=50):
    """Local search: try swapping elements or adding elements to improve the set size."""
    best_len = len(selected)
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
            # Randomize order of removal to escape local minima
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
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
                        if x != removed:
                            selected = sel1
                            prods = prods1
                            improved = True
                            break
                if improved:
                    break
            
        if not improved:
            # 3. Try swapping: remove one, add two (net +1)
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                remaining = selected[:i] + selected[i+1:]
                rem_prods = _subset_products_set(remaining)
                
                added_any = False
                # Limit candidates to speed up
                cand_subset = random.sample(candidates, min(len(candidates), 20))
                for j in range(len(cand_subset)):
                    if time.time() > deadline:
                        break
                    x1 = cand_subset[j]
                    ok1, sel1, prods1 = _try_add(rem_prods, rem_prods, x1) # Wait, _try_add takes selected list too
                    # Correction: _try_add(remaining, rem_prods, x1)
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x1)
                    if not ok1:
                        continue
                    for k in range(j+1, len(cand_subset)):
                        if time.time() > deadline:
                            break
                        x2 = cand_subset[k]
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
            # Try double swap: remove two, add two
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                for j in indices:
                    if i == j: continue
                    if time.time() > deadline:
                        break
                    remaining = [x for idx, x in enumerate(selected) if idx != i and idx != j]
                    rem_prods = _subset_products_set(remaining)
                    
                    added_any = False
                    cand_subset = random.sample(candidates, min(len(candidates), 10))
                    for x1 in cand_subset:
                        if time.time() > deadline:
                            break
                        ok1, sel1, prods1 = _try_add(remaining, rem_prods, x1)
                        if not ok1: continue
                        for x2 in cand_subset:
                            if x1 == x2: continue
                            if time.time() > deadline:
                                break
                            ok2, sel2, prods2 = _try_add(sel1, prods1, x2)
                            if ok2 and len(sel2) > len(selected):
                                selected = sel2
                                prods = prods2
                                improved = True
                                added_any = True
                                break
                        if added_any: break
                    if improved: break
                if improved: break

        if not improved:
            break
            
    return selected, prods

def construct():
    if time.time() > _DEADLINE:
        return [2]

    best_selected = []
    best_prods = set()
    
    # Run multiple restarts
    start_set = _construct_initial_set()
    
    for restart in range(20):
        if time.time() > _DEADLINE - 10:
            break
            
        if restart == 0:
            selected = start_set[:]
        else:
            # Perturb the start set
            selected = start_set[:]
            for _ in range(5):
                if len(selected) > 10:
                    i = random.randint(0, len(selected)-1)
                    selected.pop(i)
                    used = set(selected)
                    candidates = [c for c in range(2, 41) if c not in used]
                    if candidates:
                        selected.append(random.choice(candidates))
        
        prods = _subset_products_set(selected)
        
        # Refine
        if time.time() < _DEADLINE:
            selected, prods = _local_search(selected, prods, _DEADLINE)
        
        if len(selected) > len(best_selected):
            best_selected = selected
            best_prods = prods
        
        if len(best_selected) >= 18:
            break

    # Ensure validity
    if best_selected:
        final_prods = _subset_products_set(best_selected)
        if len(final_prods) != 2 ** len(best_selected):
            best_selected = start_set[:]
    
    # Fallback
    if not best_selected:
        best_selected = start_set[:]

    # Deduplicate and filter
    seen = set()
    out = []
    for x in best_selected:
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            out.append(int(x))
            
    if not out:
        out = [2]
        
    return out
