"""Improved program: structured constructive search for 9 pairwise-coprime sums.

Strategy:
1. Parity constraint forces one set all-odd, other all-even (or vice versa)
2. All 9 sums must be odd and pairwise coprime
3. Constructively find 9 pairwise coprime odd numbers that can be expressed
   as a 3x3 sum grid, then solve for A and B
4. Use bitmask precomputation for fast pairwise-coprime checks
5. Multiple search phases: constructive, targeted enumeration, randomized
"""
import itertools
import random
import time
import math

# ---- deadline guard -------------------------------------------------------
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0   # safety margin before the 90s kill

# ---- prime factor masks (odd primes only) --------------------------------
_MAX = 256
_ODD_PRIMES = [p for p in range(3, _MAX + 1)
               if all(p % q for q in range(3, int(p ** 0.5) + 1, 2))]
_num_primes = len(_ODD_PRIMES)
_mask = [0] * (_MAX + 1)
for i, p in enumerate(_ODD_PRIMES):
    for m in range(p, _MAX + 1, p):
        _mask[m] |= (1 << i)

# Precompute gcd table for fast checks
_gcd_table = {}
def _gcd(a, b):
    key = (min(a,b), max(a,b))
    if key not in _gcd_table:
        _gcd_table[key] = math.gcd(a, b)
    return _gcd_table[key]


def _pairwise_coprime(vals):
    """True if every pair of distinct values in vals is coprime."""
    seen = set()
    for v in vals:
        if v in seen:
            continue
        for u in seen:
            if _gcd(u, v) != 1:
                return False
        seen.add(v)
    return True


def _valid(A, B):
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    # pairwise coprime among distinct values
    d = list(distinct)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if _gcd(d[i], d[j]) != 1:
                return False
    return True


def _score(A, B):
    return len({a + b for a in A for b in B})


def _masks_disjoint_fast(sums):
    """Fast check using precomputed masks."""
    acc = 0
    for s in sums:
        if s < 1 or s > _MAX:
            return False
        if acc & _mask[s]:
            return False
        acc |= _mask[s]
    return True


def construct():
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    best = ([[1, 7, 13], [4, 10, 16]], 5)  # fallback seed

    def consider(A, B):
        nonlocal best
        if not _valid(A, B):
            return False
        s = _score(A, B)
        if s > best[1]:
            best = ([list(A), list(B)], s)
            if best[1] >= 9:
                return True
        return False

    # Phase 1: Constructive search - try to find 9 pairwise coprime odd sums
    # that form a valid 3x3 grid, then solve for A and B
    
    # Generate candidate odd numbers that are likely to be pairwise coprime
    # Prefer primes and numbers with large prime factors
    candidate_sums = []
    for s in range(3, 257, 2):  # odd numbers
        if s <= 256:
            candidate_sums.append(s)
    
    # Sort by "coprimeness potential" - prefer numbers with few small prime factors
    def coprimeness_score(s):
        score = 0
        for p in _ODD_PRIMES[:10]:  # Check first 10 odd primes
            if s % p == 0:
                score -= 1
        # Prefer larger numbers (more likely to be prime or have large factors)
        score += s // 10
        return score
    
    candidate_sums.sort(key=coprimeness_score, reverse=True)
    
    # Try to build a set of 9 pairwise coprime sums
    target_sums = []
    for s in candidate_sums:
        if len(target_sums) >= 9:
            break
        # Check if s is coprime to all existing target sums
        compatible = True
        for t in target_sums:
            if _gcd(s, t) != 1:
                compatible = False
                break
        if compatible:
            target_sums.append(s)
    
    # Now try to find A and B such that A+B contains these 9 sums
    # A = [a1, a2, a3], B = [b1, b2, b3]
    # Sums are: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    
    # Try different assignments of target sums to grid positions
    # For efficiency, try a few structured patterns
    
    found = False
    rng = random.Random(12345)
    
    # Phase 1a: Try structured patterns
    # Pattern: sums[i][j] = target_sums[i*3+j]
    # We need: a_i + b_j = target_sums[i*3+j] for all i,j
    # This gives us: a_i = target_sums[i*3] - b_0, and b_j = target_sums[j] - a_0
    
    # Try different base values
    for a0 in range(1, 129, 2):  # a0 must be odd
        if time.time() >= _DEADLINE:
            break
        for b0 in range(2, 129, 2):  # b0 must be even
            if time.time() >= _DEADLINE:
                break
            
            # Compute A and B from the grid
            A = [a0]
            B = [b0]
            
            # a1 = target_sums[3] - b0
            a1 = target_sums[3] - b0
            # a2 = target_sums[6] - b0
            a2 = target_sums[6] - b0
            # b1 = target_sums[1] - a0
            b1 = target_sums[1] - a0
            # b2 = target_sums[2] - a0
            b2 = target_sums[2] - a0
            
            A = [a0, a1, a2]
            B = [b0, b1, b2]
            
            # Check validity
            if (all(1 <= x <= 128 for x in A) and 
                all(1 <= x <= 128 for x in B) and
                len(set(A)) == 3 and len(set(B)) == 3 and
                all(x % 2 == 1 for x in A) and all(x % 2 == 0 for x in B)):
                sums = [a + b for a in A for b in B]
                if len(set(sums)) == 9 and _masks_disjoint_fast(sums):
                    if _valid(A, B):
                        best = ([list(A), list(B)], 9)
                        found = True
                        break
        if found:
            break
    
    # Phase 1b: If not found, try randomized search with better heuristics
    if not found and time.time() < _DEADLINE:
        # Use the target_sums as a guide - prefer sums that are in our candidate set
        # Randomized search with bias toward promising values
        
        # Create weighted pools
        odd_weights = [1 + (100 - x) // 10 for x in odds]  # Smaller odds slightly preferred
        even_weights = [1 + (100 - x) // 10 for x in evens]
        
        iters = 0
        while time.time() < _DEADLINE and not found and iters < 50000:
            iters += 1
            
            # Sample with replacement to allow bias
            A = []
            for _ in range(3):
                r = rng.random()
                # Weighted sample
                total_w = sum(odd_weights[:int(len(odd_weights)*r)] + odd_weights[int(len(odd_weights)*r):])
                # Simplified: just use uniform for now, but check against target sums
                A.append(rng.choice(odds))
            if len(set(A)) < 3:
                continue
            
            B = []
            for _ in range(3):
                B.append(rng.choice(evens))
            if len(set(B)) < 3:
                continue
            
            sums = [a + b for a in A for b in B]
            if len(set(sums)) < 9:
                continue
            if _masks_disjoint_fast(sums):
                if _valid(A, B):
                    best = ([list(A), list(B)], 9)
                    found = True
                    break
    
    # Phase 2: Focused deterministic sweep if still not found
    if not found and time.time() < _DEADLINE:
        # Use smaller pools for exhaustive search
        small_odd = [x for x in odds if x <= 50]
        small_even = [x for x in evens if x <= 50]
        
        for A in itertools.combinations(small_odd, 3):
            if time.time() >= _DEADLINE:
                break
            for B in itertools.combinations(small_even, 3):
                sums = [a + b for a in A for b in B]
                if len(set(sums)) == 9 and _masks_disjoint_fast(sums):
                    if _valid(A, B):
                        best = ([list(A), list(B)], 9)
                        found = True
                        break
            if found or time.time() >= _DEADLINE:
                break
    
    # Phase 3: If still not found, try swapped parity (A even, B odd)
    if not found and time.time() < _DEADLINE:
        rng2 = random.Random(54321)
        iters = 0
        while time.time() < _DEADLINE and not found and iters < 30000:
            iters += 1
            A = rng2.sample(evens, 3)
            B = rng2.sample(odds, 3)
            sums = [a + b for a in A for b in B]
            if len(set(sums)) == 9 and _masks_disjoint_fast(sums):
                if _valid(A, B):
                    best = ([list(A), list(B)], 9)
                    found = True
                    break
    
    return best[0]
