"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(num):
    """Return dict of prime->exponent for num."""
    factors = {}
    x = num
    for p in _PRIMES:
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[p] = cnt
    return factors

# Precompute factorizations
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _has_distinct_subset_sums(multiset):
    """Check if all subset sums of the multiset are distinct."""
    k = len(multiset)
    if k == 0:
        return True
    sums = set()
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += multiset[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _can_add(exps, facts):
    """Check if adding facts to exps keeps all subset sums distinct per prime."""
    for p in _PRIMES:
        e = facts.get(p, 0)
        if e == 0:
            continue
        new_list = exps[p] + [e]
        if not _has_distinct_subset_sums(new_list):
            return False
    return True

def _greedy_build(order):
    """Greedy construction with given candidate ordering."""
    current = list(_PRIMES)
    used = set(current)
    exps = {p: [] for p in _PRIMES}
    for p in _PRIMES:
        exps[p].append(1)
    
    for cand in order:
        if len(current) >= 18:
            break
        if cand in used:
            continue
        facts = _FACTS[cand]
        if _can_add(exps, facts):
            current.append(cand)
            used.add(cand)
            for p, e in facts.items():
                exps[p].append(e)
    return current

def _backtracking_search(start_set, remaining_candidates, target_size, deadline):
    """Backtracking to try to reach target_size."""
    if len(start_set) >= target_size:
        return start_set[:]
    if time.time() > deadline:
        return start_set[:]
    if not remaining_candidates:
        return start_set[:]
    
    # Compute exps for start_set
    exps = {p: [] for p in _PRIMES}
    for num in start_set:
        for p, e in _FACTS[num].items():
            exps[p].append(e)
    
    def _bt(current, exps, rem, depth):
        if len(current) >= target_size:
            return current[:]
        if time.time() > deadline:
            return None
        if not rem:
            return None
        for i, cand in enumerate(rem):
            facts = _FACTS[cand]
            if _can_add(exps, facts):
                new_exps = {p: exps[p][:] for p in _PRIMES}
                for p, e in facts.items():
                    new_exps[p].append(e)
                result = _bt(current + [cand], new_exps, rem[:i] + rem[i+1:], depth+1)
                if result is not None:
                    return result
        return None
    
    return _bt(start_set[:], exps, remaining_candidates, 0)

def construct():
    deadline = time.time() + 50
    
    # Try multiple greedy orderings
    candidates = [n for n in range(4, 41) if n not in _PRIMES]
    
    # Different orderings to try
    orderings = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        sorted(candidates, key=lambda x: len(_FACTS[x])),  # fewest prime factors first
        sorted(candidates, key=lambda x: -len(_FACTS[x])),  # most prime factors first
        sorted(candidates, key=lambda x: x * len(_FACTS[x])),
        sorted(candidates, key=lambda x: x // max(1, len(_FACTS[x]))),
    ]
    
    best = []
    for order in orderings:
        if time.time() > deadline - 5:
            break
        result = _greedy_build(order)
        if len(result) > len(best):
            best = result
        if len(best) >= 18:
            break
    
    # If we didn't reach 18, try backtracking from the best greedy result
    if len(best) < 18 and time.time() < deadline - 5:
        remaining = [n for n in range(2, 41) if n not in best]
        # Limit backtracking depth by only trying to add a few more
        bt_result = _backtracking_search(best, remaining, 18, deadline - 1)
        if bt_result is not None and len(bt_result) > len(best):
            best = bt_result
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
