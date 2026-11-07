import time
import random
from math import gcd

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Evaluator kills at 90s; 5s safety margin

# ---------- Prime Exponent Structure ----------
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PN = len(_PRIMES)
_EXP = {}          # n -> tuple of exponents
_EXP_LIST = {}     # n -> list of exponents

def _build_exp_table():
    for n in range(2, 41):
        m = n
        vec = [0] * _PN
        for i, p in enumerate(_PRIMES):
            while m % p == 0:
                vec[i] += 1
                m //= p
        _EXP[n] = tuple(vec)
        _EXP_LIST[n] = vec

_build_exp_table()

# ---------- Conflict Graph Pre-computation ----------
# Two numbers a and b are in conflict if their exponent vectors are rationally dependent.
# This is a necessary condition for them to cause a collision (if they are dependent, they CAN'T be in the set).
# If they are NOT dependent, they MIGHT still cause a collision with other numbers, so we still need the full check.
# But if they ARE dependent, we can immediately reject adding one if the other is present.

def _are_rationally_dependent(a, b):
    va = _EXP_LIST[a]
    vb = _EXP_LIST[b]
    # Find first non-zero index for each
    ia = -1
    for i in range(_PN):
        if va[i] != 0:
            ia = i
            break
    ib = -1
    for i in range(_PN):
        if vb[i] != 0:
            ib = i
            break
            
    if ia == -1 or ib == -1:
        return False # One is 1? No, range is 2-40. Both have exponents.
        
    # Check if va is a scalar multiple of vb
    # va[i] * vb[ia] == vb[i] * va[ia] for all i
    # We use ia as the pivot for va and ib for vb? 
    # Actually, just check proportionality using the first non-zero of va and vb.
    # Let's use the first non-zero of va (ia) and first non-zero of vb (ib).
    # If ia != ib, they are not dependent? 
    # Example: 2 (1,0) and 3 (0,1). ia=0, ib=1. Not dependent.
    # Example: 4 (2,0) and 8 (3,0). ia=0, ib=0. Dependent.
    # Example: 6 (1,1) and 12 (2,1). 
    #   va=[1,1], vb=[2,1].
    #   ia=0, ib=0.
    #   va[0]*vb[0] = 1*2 = 2. vb[0]*va[0] = 2.
    #   va[1]*vb[0] = 1*2 = 2. vb[1]*va[0] = 1*1 = 1. 2!=1. Not dependent.
    #   Correct, 6 and 12 are not rationally dependent in the sense of being powers of the same base. 
    #   Wait, 12 = 2 * 6. So 6*6 = 36, 12*1=12. No direct power relation.
    #   Rational dependence means $v_a = k v_b$ for some rational $k$.
    
    if ia != ib:
        # If the first non-zero components are at different indices, 
        # can they still be dependent? 
        # e.g. v_a = [1, 2], v_b = [2, 4]. ia=0, ib=0.
        # e.g. v_a = [0, 1], v_b = [0, 2]. ia=1, ib=1.
        # If ia != ib, say ia=0, ib=1. 
        # v_a[0] != 0, v_a[1]... 
        # v_b[0] = 0, v_b[1] != 0.
        # For v_a = k v_b, v_a[0] = k v_b[0] = 0. But v_a[0] != 0. Contradiction.
        # So if ia != ib, they are NOT dependent.
        return False

    # ia == ib
    # Check proportionality
    pivot_a = va[ia]
    pivot_b = vb[ib]
    
    for i in range(_PN):
        if va[i] * pivot_b != vb[i] * pivot_a:
            return False
    return True

# Precompute conflict pairs
# conflict[a] is a set of b such that a and b are rationally dependent
_CONFLICT = {n: set() for n in range(2, 41)}
for a in range(2, 41):
    for b in range(a + 1, 41):
        if _are_rationally_dependent(a, b):
            _CONFLICT[a].add(b)
            _CONFLICT[b].add(a)

# ---------- Core Validity Machinery ----------

def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct."""
    # Optimization: if x is in prods, return False (covered by q=x, q//x=1)
    # Standard check
    for q in prods:
        if q % x == 0:
            if (q // x) in prods:
                return False
    return True

def _try_add(selected, prods, x):
    if not _can_add(prods, x):
        return False, selected, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, selected + [x], new_prods

# ---------- Construction ----------

def _construct_initial_set():
    """Pure-powers heuristic (known valid 16-set)."""
    return [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]

def _greedy_construct(order):
    selected = []
    prods = {1}
    for x in order:
        if len(selected) >= 18:
            break
        # Fast check: is x in conflict with any selected?
        if any(_CONFLICT[x].issubset(set(selected)) or (x in _CONFLICT and selected[i] in _CONFLICT[x]) for i in range(len(selected))):
             # Wait, _CONFLICT[x] is a set of numbers. 
             # If any s in selected is in _CONFLICT[x], they are dependent.
             # If they are dependent, _can_add will fail. 
             # So we can skip the expensive _can_add.
             pass
        else:
            if _can_add(prods, x):
                selected.append(x)
                new_prods = set(prods)
                for p in prods:
                    new_prods.add(p * x)
                prods = new_prods
    return selected, prods

def _construct_random_initial_set(rng):
    selected = []
    prods = {1}
    candidates = list(range(2, 41))
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= 18:
            break
        # Fast conflict check
        conflict = False
        for s in selected:
            if s in _CONFLICT[x]:
                conflict = True
                break
        if conflict:
            continue
        if _can_add(prods, x):
            selected.append(x)
            new_prods = set(prods)
            for p in prods:
                new_prods.add(p * x)
            prods = new_prods
    return selected, prods

# ---------- Local Search ----------

def _local_search(selected, prods, deadline, max_rounds=20):
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        improved = False
        used = set(selected)
        candidates = [c for c in range(2, 41) if c not in used]
        
        # Sort candidates by how many conflicts they have with the current set?
        # Or just try all. The conflict check inside _try_add is fast if we pre-check.
        
        # 1. Try adding any candidate
        # Filter candidates that are in conflict with existing set to save time
        valid_candidates = []
        for x in candidates:
            # Check if x conflicts with any selected
            is_conflict = False
            for s in selected:
                if s in _CONFLICT[x]:
                    is_conflict = True
                    break
            if not is_conflict:
                valid_candidates.append(x)
                
        # Shuffle or sort? Random order might be better for diversity, but sorted by "conflict-free-ness" is same here.
        # Let's just iterate.
        for x in valid_candidates:
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
            # 2. Single swaps: remove one, add one different element
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i + 1:]
                rem_prods = _subset_products_set(remaining)
                
                # Candidates for swap: not in remaining, not removed (well, removed is not in remaining)
                # Filter by conflict with remaining
                potential_swaps = []
                for x in range(2, 41):
                    if x in used:
                        continue
                    # Check conflict with remaining
                    is_conflict = False
                    for s in remaining:
                        if s in _CONFLICT[x]:
                            is_conflict = True
                            break
                    if not is_conflict:
                        potential_swaps.append(x)
                        
                for x in potential_swaps:
                    if time.time() > deadline:
                        break
                    if x == removed:
                        continue
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x)
                    if ok1:
                        selected = sel1
                        prods = prods1
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            # 3. Remove one, add two (net +1)
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                remaining = selected[:i] + selected[i + 1:]
                rem_prods = _subset_products_set(remaining)
                added_any = False
                
                # Get valid candidates for the first addition
                valid_cands = []
                for x in range(2, 41):
                    if x in used:
                        continue
                    is_conflict = False
                    for s in remaining:
                        if s in _CONFLICT[x]:
                            is_conflict = True
                            break
                    if not is_conflict:
                        valid_cands.append(x)
                        
                for j in range(len(valid_cands)):
                    if time.time() > deadline:
                        break
                    x1 = valid_cands[j]
                    ok1, sel1, prods1 = _try_add(remaining, rem_prods, x1)
                    if not ok1:
                        continue
                    
                    # Now try to add x2 to sel1
                    # sel1 = remaining + [x1]
                    # Candidates for x2: not in sel1
                    for k in range(len(valid_cands)):
                        if time.time() > deadline:
                            break
                        x2 = valid_cands[k]
                        if x2 == x1:
                            continue
                        if x2 in sel1:
                            continue
                        # Check conflict of x2 with sel1
                        is_conflict = False
                        for s in sel1:
                            if s in _CONFLICT[x2]:
                                is_conflict = True
                                break
                        if is_conflict:
                            continue
                            
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

def _greedy_fill(selected, prods, deadline):
    used = set(selected)
    for x in range(2, 41):
        if time.time() > deadline:
            break
        if x in used:
            continue
        if len(selected) >= 18:
            break
        # Fast conflict check
        is_conflict = False
        for s in selected:
            if s in _CONFLICT[x]:
                is_conflict = True
                break
        if is_conflict:
            continue
            
        ok, selected, prods = _try_add(selected, prods, x)
        if ok:
            used.add(x)
    return selected, prods

def construct():
    if time.time() > _DEADLINE:
        return [2]

    best_selected = None
    best_len = 0

    # Phase 1: Strong heuristic set + local search
    selected = _construct_initial_set()
    prods = _subset_products_set(selected)
    if len(selected) < 18 and time.time() < _DEADLINE:
        selected, prods = _local_search(selected, prods, _DEADLINE)
    if time.time() < _DEADLINE and len(selected) < 18:
        selected, prods = _greedy_fill(selected, prods, _DEADLINE)
    if len(selected) > best_len:
        best_len = len(selected)
        best_selected = list(selected)

    # Phase 2: Random restarts
    rng = random.Random(42)
    remaining = _DEADLINE - time.time()
    # Allocate time for restarts
    num_restarts = max(1, int(remaining / 4.0))
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 2.0:
            break
        rsel, rprods = _construct_random_initial_set(rng)
        if len(rsel) >= 18:
            best_len = 18
            best_selected = list(rsel)
            break
        if time.time() < _DEADLINE - 1.0:
            rsel, rprods = _local_search(rsel, rprods, _DEADLINE - 1.0, max_rounds=10)
        if time.time() < _DEADLINE - 0.5 and len(rsel) < 18:
            rsel, rprods = _greedy_fill(rsel, rprods, _DEADLINE - 0.5)
        if len(rsel) > best_len:
            best_len = len(rsel)
            best_selected = list(rsel)
            if best_len >= 18:
                break

    # Ensure validity and type correctness
    selected = best_selected
    final_prods = _subset_products_set(selected)
    if len(final_prods) != 2 ** len(selected):
        # Fallback
        selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    seen = set()
    out = []
    for x in selected:
        if 2 <= x <= 40 and x not in seen:
            seen.add(x)
            out.append(int(x))

    if not out:
        out = [2]

    return out
