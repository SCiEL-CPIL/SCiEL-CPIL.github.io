"""Maximum pairwise-coprime moduli selection via greedy + local search."""
import math
import time
import random

_MAX_MOD = 128
_DEADLINE_S = 70.0


def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _is_compatible(m1, m2):
    """Two moduli are compatible (disjoint residue classes possible) iff gcd(m1,m2)==1."""
    return math.gcd(m1, m2) == 1


def _greedy(moduli, prime_factor_cache, order):
    """Greedy: add moduli in given order if compatible with all selected."""
    selected = []
    selected_factors = set()
    for m in order:
        pf = prime_factor_cache[m]
        if not (pf & selected_factors):
            selected.append(m)
            selected_factors |= pf
    return selected


def _local_search(selected, moduli, prime_factor_cache, deadline):
    """Try 1-for-1 and 1-for-2 swaps to improve."""
    selected_set = set(selected)
    selected_factors = set()
    for m in selected:
        selected_factors |= prime_factor_cache[m]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(selected)
        # Try removing one and adding two
        for i, m_remove in enumerate(selected):
            if time.time() >= deadline:
                break
            # Temporarily remove
            temp_factors = set()
            for m in selected:
                if m != m_remove:
                    temp_factors |= prime_factor_cache[m]
            
            # Find candidates that can replace
            candidates = []
            for m_add in moduli:
                if m_add in selected_set:
                    continue
                pf = prime_factor_cache[m_add]
                if not (pf & temp_factors):
                    candidates.append(m_add)
            
            # Try pairs from candidates
            best_pair = None
            best_gain = 0
            for a in range(len(candidates)):
                for b in range(a + 1, len(candidates)):
                    pa = prime_factor_cache[candidates[a]]
                    pb = prime_factor_cache[candidates[b]]
                    if pa & pb:
                        continue
                    if pa & temp_factors or pb & temp_factors:
                        continue
                    gain = 2 - 1  # net gain of 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (candidates[a], candidates[b])
            
            if best_pair:
                selected.remove(m_remove)
                selected.append(best_pair[0])
                selected.append(best_pair[1])
                selected_set.discard(m_remove)
                selected_set.add(best_pair[0])
                selected_set.add(best_pair[1])
                # Recompute factors
                selected_factors = set()
                for m in selected:
                    selected_factors |= prime_factor_cache[m]
                improved = True
    
    return selected


def construct():
    start = time.time()
    deadline = start + _DEADLINE_S
    
    moduli = list(range(2, _MAX_MOD + 1))
    n = len(moduli)
    
    # Precompute prime factors
    prime_factor_cache = {m: _prime_factors(m) for m in moduli}
    
    # Multiple greedy orderings
    best = []
    
    # Ordering 1: by number of prime factors (fewer first)
    order1 = sorted(moduli, key=lambda m: len(prime_factor_cache[m]))
    result1 = _greedy(moduli, prime_factor_cache, order1)
    best = max(best, result1, key=len)
    
    # Ordering 2: by smallest prime factor (larger first)
    order2 = sorted(moduli, key=lambda m: min(prime_factor_cache[m]), reverse=True)
    result2 = _greedy(moduli, prime_factor_cache, order2)
    best = max(best, result2, key=len)
    
    # Ordering 3: by value (larger first)
    order3 = sorted(moduli, reverse=True)
    result3 = _greedy(moduli, prime_factor_cache, order3)
    best = max(best, result3, key=len)
    
    # Ordering 4: primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    order4 = sorted(moduli, key=lambda m: (0 if is_prime(m) else 1, -m))
    result4 = _greedy(moduli, prime_factor_cache, order4)
    best = max(best, result4, key=len)
    
    # Local search on best
    if time.time() < deadline:
        best = _local_search(best, moduli, prime_factor_cache, deadline)
    
    # Try multiple random orderings
    random.seed(42)
    for _ in range(50):
        if time.time() >= deadline:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        result_r = _greedy(moduli, prime_factor_cache, random_order)
        if len(result_r) > len(best):
            best = result_r
            if time.time() < deadline:
                best = _local_search(best, moduli, prime_factor_cache, deadline)
    
    # Convert to result format
    result = [[0, m] for m in best]
    
    if not result:
        result = [[0, 2]]
    
    return result
