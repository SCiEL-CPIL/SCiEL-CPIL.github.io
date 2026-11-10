"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _factorize(num):
    """Return tuple of prime->exponent for num (indexed by _PRIMES)."""
    factors = [0] * _NUM_PRIMES
    x = num
    for i, p in enumerate(_PRIMES):
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[i] = cnt
    return tuple(factors)

# Precompute factorizations as tuples
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _can_add_to_sums(sums_set, v):
    """Check if vector v can be added to sums_set without creating duplicates.
    Returns True if safe, False if collision detected."""
    for s in sums_set:
        ns = (s[0]+v[0], s[1]+v[1], s[2]+v[2], s[3]+v[3], s[4]+v[4],
              s[5]+v[5], s[6]+v[6], s[7]+v[7], s[8]+v[8], s[9]+v[9],
              s[10]+v[10], s[11]+v[11])
        if ns in sums_set:
            return False
    return True

def _add_to_sums(sums_set, v):
    """Add all sums with v to sums_set in-place. Assumes _can_add_to_sums passed."""
    new_sums = set()
    for s in sums_set:
        ns = (s[0]+v[0], s[1]+v[1], s[2]+v[2], s[3]+v[3], s[4]+v[4],
              s[5]+v[5], s[6]+v[6], s[7]+v[7], s[8]+v[8], s[9]+v[9],
              s[10]+v[10], s[11]+v[11])
        new_sums.add(ns)
    sums_set.update(new_sums)

def _greedy_build_fast():
    """Fast greedy construction using incremental subset-sum tracking."""
    # Start with all 12 primes in [2,40]
    primes_in_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    current = list(primes_in_range)
    used = set(current)
    
    # Build initial sums set
    sums_set = {(0,0,0,0,0,0,0,0,0,0,0,0)}
    for p in primes_in_range:
        v = _FACTS[p]
        _add_to_sums(sums_set, v)
    
    # Candidates: composites in [4, 40]
    candidates = [n for n in range(4, 41) if n not in used]
    
    # Heuristic: prefer numbers whose exponent vector is "sparse" (fewer nonzero components)
    # and with smaller total exponent sum
    def heuristic(n):
        v = _FACTS[n]
        nonzero = sum(1 for x in v if x > 0)
        total = sum(v)
        return (nonzero, total, n)
    
    candidates.sort(key=heuristic)
    
    for cand in candidates:
        if len(current) >= 18:
            break
        v = _FACTS[cand]
        if _can_add_to_sums(sums_set, v):
            current.append(cand)
            used.add(cand)
            _add_to_sums(sums_set, v)
    
    return current

def construct():
    deadline = time.time() + 55  # 55s safety margin before 60s limit
    
    # Fast greedy
    result = _greedy_build_fast()
    
    if len(result) >= 18:
        return [int(x) for x in result[:18]]
    
    # If greedy didn't reach 18, try a limited backtracking search
    # Use the incremental sums approach for speed
    if time.time() < deadline:
        # Try different orderings of candidates
        primes_in_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        used_primes = set(primes_in_range)
        candidates = [n for n in range(4, 41) if n not in used_primes]
        
        # Try a few different heuristics
        heuristics = [
            lambda n: (sum(1 for x in _FACTS[n] if x > 0), sum(_FACTS[n]), n),
            lambda n: (sum(_FACTS[n]), sum(1 for x in _FACTS[n] if x > 0), n),
            lambda n: (n, sum(_FACTS[n]), sum(1 for x in _FACTS[n] if x > 0)),
            lambda n: (-n, sum(_FACTS[n]), sum(1 for x in _FACTS[n] if x > 0)),
        ]
        
        best_result = result
        
        for h in heuristics:
            if time.time() >= deadline:
                break
            cands_sorted = sorted(candidates, key=h)
            current = list(primes_in_range)
            used = set(current)
            sums_set = {(0,0,0,0,0,0,0,0,0,0,0,0)}
            for p in primes_in_range:
                _add_to_sums(sums_set, _FACTS[p])
            
            for cand in cands_sorted:
                if len(current) >= 18:
                    break
                v = _FACTS[cand]
                if _can_add_to_sums(sums_set, v):
                    current.append(cand)
                    used.add(cand)
                    _add_to_sums(sums_set, v)
            
            if len(current) > len(best_result):
                best_result = current
        
        result = best_result
    
    if len(result) >= 18:
        return [int(x) for x in result[:18]]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
