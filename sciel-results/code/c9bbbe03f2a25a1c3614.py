import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 85.0


def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct.
    Division trick: p*x in prods iff exists q in prods with q%x==0 and q//x in prods."""
    for q in prods:
        if q % x == 0:
            if q // x in prods:
                return False
    return True


def _try_add(prods, x):
    if not _can_add(prods, x):
        return False, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, new_prods


def _greedy_add(base_prods, base_selected, candidates):
    """Greedily add as many candidates as possible."""
    prods = base_prods.copy()
    selected = list(base_selected)
    for x in candidates:
        if _can_add(prods, x):
            _, prods = _try_add(prods, x)
            selected.append(x)
    return selected, prods


def construct():
    if time.time() > _DEADLINE:
        return [2]

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    all_nums = list(range(2, 41))
    prime_set = set(primes)

    # Build prods for all 12 primes
    prime_prods = {1}
    for x in primes:
        _, prime_prods = _try_add(prime_prods, x)

    # Phase 1: Greedy add to full prime set
    candidates = [x for x in all_nums if x not in prime_set]
    sel1, prods1 = _greedy_add(prime_prods, primes, candidates)
    best_result = list(sel1)
    best_count = len(sel1)

    # Phase 2: Local search - remove one element, re-greedy
    if time.time() < _DEADLINE - 5:
        for i in range(len(best_result)):
            if time.time() > _DEADLINE - 2:
                break
            rem = best_result[i]
            remaining = best_result[:i] + best_result[i+1:]
            # Rebuild prods
            prods2 = {1}
            valid = True
            for x in remaining:
                if not _can_add(prods2, x):
                    valid = False
                    break
                _, prods2 = _try_add(prods2, x)
            if not valid:
                continue
            new_sel, _ = _greedy_add(prods2, remaining, [x for x in all_nums if x not in set(remaining)])
            if len(new_sel) > best_count:
                best_count = len(new_sel)
                best_result = list(new_sel)

    # Phase 3: Try dropping 1-2 primes from the prime set, then greedy
    if time.time() < _DEADLINE - 10:
        # Drop 1 prime
        for p in primes:
            if time.time() > _DEADLINE - 5:
                break
            base = [q for q in primes if q != p]
            prods_b = {1}
            for x in base:
                _, prods_b = _try_add(prods_b, x)
            cands = [x for x in all_nums if x not in set(base)]
            new_sel, _ = _greedy_add(prods_b, base, cands)
            if len(new_sel) > best_count:
                best_count = len(new_sel)
                best_result = list(new_sel)

    if time.time() < _DEADLINE - 10:
        # Drop 2 primes
        for i in range(len(primes)):
            for j in range(i+1, len(primes)):
                if time.time() > _DEADLINE - 5:
                    break
                base = [q for q in primes if q != primes[i] and q != primes[j]]
                prods_b = {1}
                for x in base:
                    _, prods_b = _try_add(prods_b, x)
                cands = [x for x in all_nums if x not in set(base)]
                new_sel, _ = _greedy_add(prods_b, base, cands)
                if len(new_sel) > best_count:
                    best_count = len(new_sel)
                    best_result = list(new_sel)

    # Phase 4: Second round of local search on best result
    if time.time() < _DEADLINE - 5 and best_count < 18:
        for i in range(len(best_result)):
            if time.time() > _DEADLINE - 2:
                break
            rem = best_result[i]
            remaining = best_result[:i] + best_result[i+1:]
            prods2 = {1}
            valid = True
            for x in remaining:
                if not _can_add(prods2, x):
                    valid = False
                    break
                _, prods2 = _try_add(prods2, x)
            if not valid:
                continue
            new_sel, _ = _greedy_add(prods2, remaining, [x for x in all_nums if x not in set(remaining)])
            if len(new_sel) > best_count:
                best_count = len(new_sel)
                best_result = list(new_sel)

    return best_result if best_count >= 1 else [2]
