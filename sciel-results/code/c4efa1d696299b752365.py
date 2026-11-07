"""Combined program: multi-order greedy + local-search refinement to maximize
distinct integers in 2..40 with all 2^k subset products distinct."""
import time
import random
from math import isqrt

_P0 = time.time()
_DEADLINE = _P0 + 85.0  # safety margin under 90s wall-clock limit

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

def _greedy_build(order, deadline):
    """Build a valid set by greedily adding elements in the given order."""
    selected = []
    prods = {1}
    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        ok, selected, prods = _try_add(selected, prods, x)
        if not ok:
            continue
    return selected, prods

def _local_search(selected, prods, deadline, max_rounds=10):
    """Local search: try various moves to improve the set size."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # Strategy 1: Try removing one element and adding one better candidate
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            remaining = selected[:i] + selected[i+1:]
            rem_prods = _subset_products_set(remaining)
            for x in candidates:
                ok, sel_new, prods_new = _try_add(remaining, rem_prods, x)
                if ok and len(sel_new) > len(selected):
                    selected = sel_new
                    prods = prods_new
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
            
        # Strategy 2: Try removing one element and adding two others
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            remaining = selected[:i] + selected[i+1:]
            rem_prods = _subset_products_set(remaining)
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
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
            
        # Strategy 3: Try removing two elements and adding three others
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        if len(selected) >= 2:
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                for m in range(i+1, len(selected)):
                    if time.time() > deadline:
                        break
                    remaining = selected[:i] + selected[i+1:m] + selected[m+1:]
                    rem_prods = _subset_products_set(remaining)
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
                            if not ok2:
                                continue
                            for l in range(k+1, len(candidates)):
                                if time.time() > deadline:
                                    break
                                x3 = candidates[l]
                                ok3, sel3, prods3 = _try_add(sel2, prods2, x3)
                                if ok3 and len(sel3) > len(selected):
                                    selected = sel3
                                    prods = prods3
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return selected, prods

def _smart_greedy_with_backtrack(order, deadline, max_backtrack=3):
    """Greedy with limited backtracking: if stuck, try skipping earlier elements."""
    selected = []
    prods = {1}
    idx = 0
    
    while idx < len(order) and time.time() < deadline and len(selected) < 18:
        x = order[idx]
        ok, selected, prods = _try_add(selected, prods, x)
        if ok:
            idx += 1
        else:
            # Try backtracking: remove last few elements and try different choices
            backtracked = False
            for bt in range(1, min(max_backtrack, len(selected)) + 1):
                if time.time() > deadline:
                    break
                if bt > len(selected):
                    break
                # Remove last bt elements
                removed = selected[-bt:]
                selected = selected[:-bt]
                prods = _subset_products_set(selected)
                
                # Try to add alternatives from remaining order
                alt_idx = idx - bt  # go back in order
                for alt_x in order[alt_idx:]:
                    if time.time() > deadline:
                        break
                    ok_alt, selected_alt, prods_alt = _try_add(selected, prods, alt_x)
                    if ok_alt and len(selected_alt) > len(selected):
                        selected = selected_alt
                        prods = prods_alt
                        idx = order.index(alt_x) + 1
                        backtracked = True
                        break
                if backtracked:
                    break
            if not backtracked:
                idx += 1
    
    return selected, prods

def construct():
    global _DEADLINE
    if time.time() > _DEADLINE:
        return [2]

    # Primes and composites in 2..40
    primes = []
    for a in range(2, 41):
        if all(a % d for d in range(2, isqrt(a) + 1)):
            primes.append(a)
    composites = [c for c in range(2, 41) if c not in primes]

    # Multiple greedy orderings
    best_selected = []
    best_prods = {1}

    # Order 1: primes first, then composites ascending
    order1 = primes + composites
    sel1, prods1 = _greedy_build(order1, _DEADLINE)
    if len(sel1) > len(best_selected):
        best_selected = sel1
        best_prods = prods1

    # Order 2: composites first, then primes
    order2 = composites + primes
    if time.time() < _DEADLINE:
        sel2, prods2 = _greedy_build(order2, _DEADLINE)
        if len(sel2) > len(best_selected):
            best_selected = sel2
            best_prods = prods2

    # Order 3: mixed - interleave primes and composites
    order3 = []
    pi = ci = 0
    while pi < len(primes) or ci < len(composites):
        if pi < len(primes) and (ci >= len(composites) or pi <= ci):
            order3.append(primes[pi])
            pi += 1
        else:
            order3.append(composites[ci])
            ci += 1
    if time.time() < _DEADLINE:
        sel3, prods3 = _greedy_build(order3, _DEADLINE)
        if len(sel3) > len(best_selected):
            best_selected = sel3
            best_prods = prods3

    # Order 4: descending (larger numbers first, may allow more small ones later)
    order4 = list(range(40, 1, -1))
    if time.time() < _DEADLINE:
        sel4, prods4 = _greedy_build(order4, _DEADLINE)
        if len(sel4) > len(best_selected):
            best_selected = sel4
            best_prods = prods4

    # Order 5: alternating small/large
    order5 = []
    small = list(range(2, 21))
    large = list(range(40, 20, -1))
    si = li = 0
    while si < len(small) or li < len(large):
        if si < len(small) and (li >= len(large) or si <= li):
            order5.append(small[si])
            si += 1
        else:
            order5.append(large[li])
            li += 1
    if time.time() < _DEADLINE:
        sel5, prods5 = _greedy_build(order5, _DEADLINE)
        if len(sel5) > len(best_selected):
            best_selected = sel5
            best_prods = prods5

    # Randomized orderings
    rng = random.Random(42)
    for _ in range(8):
        if time.time() > _DEADLINE:
            break
        order = list(range(2, 41))
        rng.shuffle(order)
        sel, prods = _greedy_build(order, _DEADLINE)
        if len(sel) > len(best_selected):
            best_selected = sel
            best_prods = prods

    # Smart greedy with backtracking on best order
    if time.time() < _DEADLINE:
        sel_bt, prods_bt = _smart_greedy_with_backtrack(order1, _DEADLINE, max_backtrack=3)
        if len(sel_bt) > len(best_selected):
            best_selected = sel_bt
            best_prods = prods_bt

    # Local search refinement on the best found so far
    if time.time() < _DEADLINE and len(best_selected) < 18:
        best_selected, best_prods = _local_search(best_selected, best_prods, _DEADLINE)

    # Final refinement: try adding any remaining candidates
    used = set(best_selected)
    if time.time() < _DEADLINE and len(best_selected) < 18:
        for x in range(2, 41):
            if x in used:
                continue
            if len(best_selected) >= 18:
                break
            ok, best_selected, best_prods = _try_add(best_selected, best_prods, x)
            if ok:
                used.add(x)

    # Guarantee validity & type correctness
    best_selected = [int(x) for x in best_selected if 2 <= x <= 40]
    # dedupe preserving order
    seen = set()
    out = []
    for x in best_selected:
        if x not in seen:
            seen.add(x)
            out.append(x)
    if len(out) < 1:
        out = [2]
    return list(out)
