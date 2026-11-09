# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

def _is_semiprime_distinct(n):
    """Check if n = p*q for distinct primes p, q (or p^2 where p is prime - but that's not distinct).
    Actually: n = p*q where p, q are distinct primes."""
    if n < 4:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            q = n // d
            if d != q and _is_prime(d) and _is_prime(q):
                return True
            # If d == q, then n = d^2, not distinct
            # We only need to find one such factorization
            # Actually, if n has any factorization into two distinct primes, return True
            # We need to be more careful: n might have multiple factorizations
            # But if n = p*q*r, it might also have p*(q*r) but q*r is not prime
            # The question is: does there exist primes p != q such that p*q = n?
            # If n has exactly two prime factors (counting multiplicity) and they're distinct, yes
            # If n has more, it might still be expressible as p*q only if one factor is composite
            # Wait: n = p*q means n has exactly two prime factors (counting multiplicity) and they're different
            # So n must be a semiprime with distinct prime factors
            return True
    return False

def _count_distinct_prime_factors(n):
    """Return the number of distinct prime factors of n."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n_max = 256
    candidates = list(range(2, n_max + 1))
    
    # Precompute prime status
    prime_flags = [False] * (n_max + 1)
    for p in range(2, n_max + 1):
        prime_flags[p] = _is_prime(p)
    
    # Identify dangerous numbers: those that are p*q for distinct primes p, q
    dangerous = set()
    primes = [p for p in range(2, n_max + 1) if prime_flags[p]]
    for i, p in enumerate(primes):
        for q in primes[i+1:]:
            pq = p * q
            if pq <= n_max:
                dangerous.add(pq)
    
    # Precompute divisibility structure
    # For each a, we want to know: which pairs (b,c) with b,c in our set have a | b*c?
    # Key insight: a | b*c iff for every prime power p^k || a, we have p^k | b*c
    # This means: for each prime p dividing a, the total exponent of p in b*c must be >= exponent in a
    
    # Strategy: Greedy construction with conflict counting
    # Start with all non-dangerous numbers
    current_set = set(x for x in candidates if x not in dangerous)
    
    # Also consider: should we include some dangerous numbers?
    # A dangerous number d = p*q where p,q are distinct primes.
    # d | b*c requires p|b*c and q|b*c.
    # If p and q are both in our set, then choosing b=p, c=q gives d|p*q, violation.
    # But if we don't have both p and q in the set, we might include d.
    # For now, let's start with non-dangerous and see.
    
    # Greedy removal to ensure validity
    def _has_conflict(s):
        """Check if set s has any violating triple. Return the triple or None."""
        lst = sorted(s)
        for i in range(len(lst)):
            a = lst[i]
            for j in range(i+1, len(lst)):
                b = lst[j]
                for k in range(j+1, len(lst)):
                    c = lst[k]
                    if (b * c) % a == 0:
                        return (a, b, c)
        return None
    
    # Iteratively remove elements that cause conflicts
    changed = True
    while changed and time.time() < deadline:
        changed = False
        conflict = _has_conflict(current_set)
        if conflict is None:
            break
        
        a, b, c = conflict
        # Decide which element to remove
        # Heuristic: remove the element that causes the most conflicts
        # Count conflicts for each of a, b, c
        def _conflict_count(elem):
            count = 0
            for x in current_set:
                if x == elem:
                    continue
                for y in current_set:
                    if y == x or y == elem:
                        continue
                    # Check if elem, x, y form a conflict in any ordering
                    # We need: one divides the product of the other two
                    if (x * y) % elem == 0:
                        count += 1
                    elif (elem * y) % x == 0:
                        count += 1
                    elif (elem * x) % y == 0:
                        count += 1
            return count
        
        # For efficiency, just pick the smallest one (a is already smallest)
        # Or remove the one with highest conflict count
        conflict_a = _conflict_count(a)
        conflict_b = _conflict_count(b)
        conflict_c = _conflict_count(c)
        
        # Remove the one with highest conflict count
        if conflict_a >= conflict_b and conflict_a >= conflict_c:
            current_set.remove(a)
        elif conflict_b >= conflict_a and conflict_b >= conflict_c:
            current_set.remove(b)
        else:
            current_set.remove(c)
        changed = True
    
    # Verify final set
    if _has_conflict(current_set) is not None:
        # Fallback: aggressive removal
        lst = sorted(current_set)
        for a in lst:
            trial = current_set - {a}
            if _has_conflict(trial) is None:
                current_set = trial
                break
    
    # Try to add back some dangerous numbers if they don't cause conflicts
    for d in sorted(dangerous):
        if d in current_set:
            continue
        trial = current_set | {d}
        if _has_conflict(trial) is None:
            current_set = trial
    
    # Final validation
    if _has_conflict(current_set) is not None:
        # Emergency: keep only primes and prime powers
        current_set = set(p for p in range(2, n_max + 1) if prime_flags[p])
        # Also add prime powers p^k <= 256
        for p in primes:
            pk = p
            while pk * p <= n_max:
                pk *= p
                current_set.add(pk)
    
    result = sorted(current_set)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
