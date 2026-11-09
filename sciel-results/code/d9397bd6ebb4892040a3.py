"""Reformulated as a gap-filling CSP: state is the set of uncovered integers,
moves select a prime+residue that maximally reduces the uncovered set, subject
to the reciprocal budget. Uses bitset for fast coverage checks."""

import time
from fractions import Fraction

# Deadline guard: return best result before 145s (5s safety margin under 150s)
_DEADLINE = time.time() + 145.0

def _is_deadline():
    return time.time() >= _DEADLINE

def _primes_up_to(n):
    """Sieve of Eratosthenes for primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _build_coverage_bitset(p, r, max_n):
    """Return a bitset (as an int) where bit i is set iff i is in residue class r mod p, for i in 1..max_n."""
    # Numbers congruent to r mod p: r, r+p, r+2p, ... but we only care about 1..max_n
    # If r == 0, the numbers are p, 2p, 3p, ...
    # If r > 0, the numbers are r, r+p, r+2p, ...
    # We want bits 1..max_n set.
    mask = 0
    if r == 0:
        start = p
    else:
        start = r
    if start < 1:
        start += p * ((1 - start + p - 1) // p)
    for x in range(start, max_n + 1, p):
        if 1 <= x <= max_n:
            mask |= (1 << x)
    return mask

def _compute_L(covered_mask, max_n):
    """Find the largest L such that bits 1..L are all set in covered_mask."""
    # Check from 1 upward
    for i in range(1, max_n + 1):
        if not (covered_mask & (1 << i)):
            return i - 1
    return max_n

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    MAX_N = 100000  # cap
    primes = _primes_up_to(1000)
    
    # Precompute coverage bitsets for each prime and a few candidate residues
    # We'll use a greedy approach: at each step, pick the (p, r) that maximally
    # reduces the uncovered set, subject to budget.
    
    # Budget: sum of 1/p <= 1, using exact rationals
    budget = Fraction(1, 1)
    used_primes = set()
    covered_mask = 0
    classes = []
    
    # For efficiency, precompute coverage masks for all (p, r) pairs
    # But that's 168 primes * up to 1000 residues = ~168000 masks of 100000 bits
    # That's too much memory. Instead, compute on the fly but cache.
    
    # Alternative: for each prime p, the residue r that best covers the current
    # uncovered set can be found by trying all r in 0..p-1.
    # For p up to 1000, that's at most 1000 tries per prime, 168 primes = 168000
    # coverage computations. Each is O(max_n/p) bit operations.
    
    # Let's use a smarter approach: represent uncovered as a set of integers,
    # but use bitset for fast intersection.
    
    # Actually, let's precompute for each prime p, an array of bitsets for each residue.
    # But 100000-bit ints are ~12.5 KB each. 168000 * 12.5 KB = 2.1 GB. Too much.
    
    # Better: compute coverage on the fly. For a given (p, r), the coverage mask
    # has bits at positions r, r+p, r+2p, ... up to MAX_N.
    # We can compute the number of newly covered elements as:
    #   newly = uncovered & coverage_mask(p, r)
    #   count = newly.bit_count()
    
    # To compute coverage_mask(p, r) efficiently without storing it:
    # We can iterate and count, but that's O(MAX_N/p) per query.
    
    # Let's try a different representation: use a sorted list of uncovered numbers
    # and for each (p, r), count how many uncovered numbers are ≡ r (mod p).
    
    # Actually, for MAX_N = 100000, and we want L >= 999, let's focus on MAX_N = 1000
    # first to get a good solution, then try to extend.
    
    # Let's use a hybrid: bitset for the coverage, but compute it incrementally.
    
    # Strategy: 
    # 1. Start with no primes used, covered = 0.
    # 2. Greedily pick (p, r) that maximizes new coverage, subject to budget.
    # 3. After greedy, do local search: try swapping residues or adding/removing primes.
    
    # For the greedy step, we need to evaluate many (p, r) pairs.
    # Let's precompute for each prime p, a list of (r, coverage_mask) for all r.
    # But store as a list of ints. 168 primes * avg 500 residues * 12.5 KB = ~1 GB. Still too much.
    
    # Alternative: don't precompute. For each candidate (p, r), compute the new coverage
    # count by iterating through multiples. For p >= 13, MAX_N/p <= 100000/13 ≈ 7700.
    # 168 primes * 500 residues * 7700 = 648 million operations. Too slow.
    
    # Let's reduce the search space. We know from the seed that using primes from 13 upward
    # gives sum of reciprocals ≈ 0.92. Let's work with that set of primes.
    
    # Key insight: for the greedy step, we can compute the "gain" of adding (p, r) as:
    #   gain = number of i in 1..MAX_N such that i ≡ r (mod p) AND i not yet covered.
    # We can maintain a bitset of covered numbers and compute:
    #   gain = popcount(uncovered & cover_mask(p, r))
    
    # To compute cover_mask(p, r) on the fly:
    def _cover_mask(p, r, max_n):
        mask = 0
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, max_n + 1, p):
            mask |= (1 << x)
        return mask
    
    # This is O(max_n/p) per call. For p=13, that's ~7700 iterations.
    # 168 primes * 13 residues (just try a few) = manageable.
    
    # Let's try: for each unused prime p, try all r in 0..p-1 (or a subset),
    # compute gain, pick the best.
    
    # To speed up, let's precompute cover masks for small primes only, and
    # for large primes, just try r = 0 or r = 1 or a few candidates.
    
    # Actually, let's just do it and see if it's fast enough.
    # 168 primes, for each try all r: sum of p for p in primes = sum of first 168 primes ≈ 600000.
    # For each (p, r), computing cover_mask is O(MAX_N/p).
    # Total: sum over p of p * (MAX_N/p) = 168 * MAX_N = 16.8 million bit sets.
    # Each bit set is a Python big-int OR, which is expensive.
    
    # Let's use a different approach: represent covered as a boolean array.
    
    covered = [False] * (MAX_N + 1)
    used = set()
    
    def _count_new(p, r):
        """Count how many numbers in 1..MAX_N are ≡ r mod p and not yet covered."""
        count = 0
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, MAX_N + 1, p):
            if not covered[x]:
                count += 1
        return count
    
    def _mark_covered(p, r):
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        for x in range(start, MAX_N + 1, p):
            covered[x] = True
    
    # Greedy: repeatedly pick the (p, r) with max gain, subject to budget.
    best_score = 0
    best_classes = []
    
    while not _is_deadline():
        best_gain = 0
        best_p = None
        best_r = None
        
        for p in primes:
            if p in used:
                continue
            if budget - Fraction(1, p) < 0:
                continue
            # Try all residues
            for r in range(p):
                gain = _count_new(p, r)
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
                    if _is_deadline():
                        break
            if _is_deadline():
                break
            # Early termination if gain is 0
            if best_gain == 0:
                break
        
        if best_p is None or best_gain == 0:
            break
        
        # Add this class
        _mark_covered(best_p, best_r)
        used.add(best_p)
        budget -= Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        # Compute current L
        L = 0
        for i in range(1, MAX_N + 1):
            if covered[i]:
                L = i
            else:
                break
        if L > best_score:
            best_score = L
            best_classes = list(classes)
        
        if L >= 999:
            break
    
    # Local search: try to improve by swapping residues
    if not _is_deadline() and best_classes:
        for _iteration in range(3):  # 3 improvement passes
            if _is_deadline():
                break
            improved = False
            for idx in range(len(best_classes)):
                if _is_deadline():
                    break
                p, r = best_classes[idx]
                # Try changing residue
                for new_r in range(p):
                    if new_r == r:
                        continue
                    # Compute gain of changing r to new_r
                    # First, unmark old coverage
                    old_covered_count = 0
                    if r == 0:
                        start_old = p
                    else:
                        start_old = r
                    if start_old < 1:
                        start_old += ((1 - start_old + p - 1) // p) * p
                    
                    # Count how many currently covered numbers would be lost
                    lost = 0
                    for x in range(start_old, MAX_N + 1, p):
                        if covered[x]:
                            lost += 1
                    
                    # Count how many new numbers would be gained
                    if new_r == 0:
                        start_new = p
                    else:
                        start_new = new_r
                    if start_new < 1:
                        start_new += ((1 - start_new + p - 1) // p) * p
                    
                    gained = 0
                    for x in range(start_new, MAX_N + 1, p):
                        if not covered[x]:
                            gained += 1
                    
                    if gained > lost:
                        # Apply change
                        # Unmark old
                        for x in range(start_old, MAX_N + 1, p):
                            if covered[x]:
                                # Check if still covered by another class
                                still_covered = False
                                for (p2, r2) in best_classes:
                                    if p2 == p:
                                        continue
                                    if (x % p2) == r2:
                                        still_covered = True
                                        break
                                if not still_covered:
                                    covered[x] = False
                        # Mark new
                        for x in range(start_new, MAX_N + 1, p):
                            covered[x] = True
                        best_classes[idx] = [p, new_r]
                        improved = True
                        break
            if not improved:
                break
    
    # Compute final L
    L = 0
    for i in range(1, MAX_N + 1):
        if covered[i]:
            L = i
        else:
            break
    
    if L > best_score:
        best_score = L
        best_classes = list(classes)
    
    # Ensure budget constraint is met
    total = sum(Fraction(1, p) for p, r in best_classes)
    if total > 1:
        # Remove classes until budget is met
        while total > 1 and best_classes:
            # Remove the class with largest 1/p (smallest p)
            best_classes.sort(key=lambda x: x[0])
            p, r = best_classes.pop(0)
            total -= Fraction(1, p)
    
    return best_classes
