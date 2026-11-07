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

def _local_search(selected, prods, deadline, max_rounds=5):
    """Local search: try swapping elements to improve the set size."""
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        # Try removing one element and adding two others
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            # Try removing selected[i]
            remaining = selected[:i] + selected[i+1:]
            rem_prods = _subset_products_set(remaining)
            # Try adding two candidates
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
        if not improved:
            break
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

    # Randomized orderings
    rng = random.Random(42)
    for _ in range(5):
        if time.time() > _DEADLINE:
            break
        order = list(range(2, 41))
        rng.shuffle(order)
        sel, prods = _greedy_build(order, _DEADLINE)
        if len(sel) > len(best_selected):
            best_selected = sel
            best_prods = prods

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
