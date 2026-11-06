"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_MAX_MOD = 128
_DEADLINE = 80.0  # seconds; leave margin under 90s kill

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    if g == 1:
        return False
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, chosen):
    for (r, m) in chosen:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True

def _greedy(order, start_time=None, deadline=None):
    """Try to add each modulus in 'order', picking a residue if possible."""
    chosen = []
    for m in order:
        if start_time is not None and time.time() - start_time > deadline:
            break
        found = False
        # Try residues in a deterministic order first, then shuffled
        for r in range(m):
            if _can_add(r, m, chosen):
                chosen.append((r, m))
                found = True
                break
        if not found:
            # Try a few random residues
            for _ in range(20):
                r = random.randrange(m)
                if _can_add(r, m, chosen):
                    chosen.append((r, m))
                    found = True
                    break
    return chosen

def _moduli_order_by_shared_factors():
    """Order moduli so that those sharing factors with many others come first."""
    moduli = list(range(2, _MAX_MOD + 1))
    # Score: number of other moduli that share a factor > 1
    scores = {}
    for m in moduli:
        s = 0
        for n in moduli:
            if m != n and _gcd(m, n) > 1:
                s += 1
        scores[m] = s
    # Process high-score first (more constrained)
    moduli.sort(key=lambda x: (-scores[x], x))
    return moduli

def _moduli_order_by_num_divisors():
    """Order by number of divisors descending (more structure first)."""
    moduli = list(range(2, _MAX_MOD + 1))
    def num_divs(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    moduli.sort(key=lambda x: (-num_divs(x), x))
    return moduli

def _moduli_order_by_max_prime_factor():
    """Order by largest prime factor descending."""
    moduli = list(range(2, _MAX_MOD + 1))
    def max_pf(n):
        pf = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                pf = max(pf, n // d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            pf = max(pf, n)
        return pf
    moduli.sort(key=lambda x: (-max_pf(x), x))
    return moduli

def construct():
    start_time = time.time()
    deadline = _DEADLINE

    # Precompute several deterministic orderings
    orderings = [
        _moduli_order_by_shared_factors(),
        _moduli_order_by_num_divisors(),
        _moduli_order_by_max_prime_factor(),
        list(range(2, _MAX_MOD + 1)),
        list(range(_MAX_MOD, 1, -1)),
    ]

    best = []
    
    # First pass: deterministic orderings
    for order in orderings:
        if time.time() - start_time > deadline - 5:
            break
        result = _greedy(order, start_time, deadline - 5)
        if len(result) > len(best):
            best = result

    # Second pass: randomized restarts with mixed orderings
    random.seed(42)
    attempts = 0
    while time.time() - start_time < deadline - 2:
        attempts += 1
        # Create a randomized ordering: mix of structured + random
        if attempts % 3 == 0:
            order = list(range(2, _MAX_MOD + 1))
            random.shuffle(order)
        elif attempts % 3 == 1:
            order = _moduli_order_by_shared_factors()
            # Perturb: swap a few elements
            for _ in range(5):
                i, j = random.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
        else:
            order = _moduli_order_by_num_divisors()
            for _ in range(5):
                i, j = random.sample(range(len(order)), 2)
                order[i], order[j] = order[j], order[i]
        
        result = _greedy(order, start_time, deadline - 2)
        if len(result) > len(best):
            best = result

    # Convert to required format: list of [residue, modulus]
    if not best:
        # Fallback: at least return something valid
        return [[1, 2]]
    
    return [[r, m] for (r, m) in best]
# EVOLVE-BLOCK-END
