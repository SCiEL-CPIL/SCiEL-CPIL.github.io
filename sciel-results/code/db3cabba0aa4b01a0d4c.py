"""
Strategy: Combinatorial Analysis of Divisor Adjacency for Structured Forms.

For n = 2^a * p_1 * p_2 * ... * p_k (distinct odd primes), we can compute
tau_perp(n) analytically by analyzing the divisor lattice structure.

Key insight: In the sorted divisors of 2^a * P where P is squarefree,
adjacent pairs (d_i, d_{i+1}) are coprime iff they differ by a factor
that doesn't share any odd prime. This happens when moving between
"layers" of 2-adic valuation in specific ways.

We optimize over (a, k) and use the smallest primes to maximize k
within the 10^18 budget.
"""
# EVOLVE-BLOCK-START
import time
import math

DEADLINE = 145.0
LIMIT = 10**18

def _sieve(n):
    """Generate primes up to n using sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

_SMALL_PRIMES = _sieve(200)

def _score_bruteforce(n):
    """
    Calculate the number of coprime adjacent pairs in the sorted divisors of n.
    Used for verification and for general n.
    """
    if n < 2:
        return 0
    
    divisors = [1]
    temp = n
    for p in _SMALL_PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
                divisors += [d * power for d in divisors]
    if temp > 1:
        divisors += [d * temp for d in divisors]
    
    divisors = sorted(set(divisors))
    
    count = 0
    for i in range(1, len(divisors)):
        if math.gcd(divisors[i-1], divisors[i]) == 1:
            count += 1
    return count

def _score_structured(a, k):
    """
    Compute tau_perp for n = 2^a * p_1 * ... * p_k (smallest k odd primes).
    
    The divisors are 2^i * d where 0 <= i <= a and d | P (P = product of k primes).
    Sorted order: group by 2-adic valuation. Within each valuation level,
    the odd divisors are sorted.
    
    Adjacent coprime pairs occur when:
    1. Moving from 2^i * d to 2^i * d' where gcd(d, d') = 1 (within same level)
    2. Moving from 2^i * d to 2^(i+1) * d' where gcd(d, 2*d') = 1, i.e., d is odd and gcd(d, d')=1
    
    Actually, let's think more carefully. The sorted divisors of 2^a * P:
    Level 0: 1, p1, p2, p1*p2, ... (all divisors of P, sorted)
    Level 1: 2, 2*p1, 2*p2, ... (2 * divisors of P, sorted)
    ...
    Level a: 2^a, 2^a*p1, ... (2^a * divisors of P, sorted)
    
    Within a level, consecutive divisors 2^i * d and 2^i * d' are coprime
    iff gcd(d, d') = 1 (since 2^i is even for i>=1, but gcd(2^i*d, 2^i*d') = 2^i * gcd(d,d'),
    so they're coprime iff i=0 and gcd(d,d')=1... wait that's not right either.
    
    gcd(2^i * d, 2^i * d') = 2^i * gcd(d, d'). For this to be 1, we need i=0 and gcd(d,d')=1.
    
    Between levels: 2^i * d (last in level i) and 2^(i+1) * d' (first in level i+1).
    gcd(2^i * d, 2^(i+1) * d') = 2^i * gcd(d, 2*d') = 2^i * gcd(d, 2) * gcd(d, d').
    For i >= 1, this has factor 2^i, so not 1.
    For i = 0: gcd(d, 2*d') = gcd(d, 2) * gcd(d, d'). Since d is odd (divisor of P), gcd(d,2)=1,
    so we need gcd(d, d') = 1.
    
    So coprime adjacent pairs only occur:
    - Within level 0 (i=0): consecutive odd divisors that are coprime
    - Between level 0 and level 1: last odd divisor and first even divisor (2 * smallest odd divisor = 2)
      Wait, the last in level 0 is the largest divisor of P, and first in level 1 is 2.
      gcd(largest_odd_divisor, 2) = 1 since largest is odd. So this pair is always coprime!
    
    Let me reconsider. Within level i (i >= 1): divisors are 2^i * d where d | P.
    gcd(2^i * d, 2^i * d') = 2^i * gcd(d, d'). This equals 1 only if i=0.
    
    So for i >= 1, NO adjacent pairs within the level are coprime.
    
    Between level i and level i+1 (i >= 0):
    Last in level i: 2^i * max(P) = 2^i * P
    First in level i+1: 2^(i+1) * 1 = 2^(i+1)
    gcd(2^i * P, 2^(i+1)) = 2^i * gcd(P, 2) = 2^i * 1 = 2^i.
    This is 1 only if i = 0.
    
    So the only coprime adjacent pairs are:
    1. Within level 0: consecutive divisors of P that are coprime
    2. The pair (P, 2) between level 0 and level 1 (if a >= 1)
    
    Wait, I need to be more careful about the ordering. The divisors are sorted numerically.
    Level 0: 1, p1, p2, p1*p2, ..., P (sorted)
    Level 1: 2, 2*p1, 2*p2, ..., 2*P (sorted)
    
    The largest in level 0 is P. The smallest in level 1 is 2.
    Is P < 2? Only if P = 1, which means k=0. For k >= 1, P >= 3 > 2.
    So the ordering is NOT simply all of level 0 then all of level 1!
    
    This is the key issue. The divisors are interleaved.
    
    Let me reconsider. For n = 2^a * P, the divisors sorted:
    1, 2, p1, 4, 2*p1, p2, ... it's complex.
    
    I'll use a different approach: for small a and k, compute directly but efficiently.
    """
    # For now, fall back to brute force for correctness
    # Build n
    if k == 0:
        n = 2 ** a
    else:
        prod = 1
        for i in range(k):
            prod *= _SMALL_PRIMES[i + 1]  # skip 2, use odd primes
        n = (2 ** a) * prod
    
    if n > LIMIT:
        return -1
    
    return _score_bruteforce(n)

def construct():
    start = time.time()
    best_n = 2
    best_s = 0
    
    def consider(n, s):
        nonlocal best_n, best_s
        if 2 <= n <= LIMIT and s > best_s:
            best_s = s
            best_n = n
    
    # Phase 1: Try n = 2^a * (product of k smallest odd primes)
    # for various a and k
    for a in range(1, 61):  # 2^a <= 10^18 => a <= 59
        if 2**a > LIMIT:
            break
        remaining = LIMIT // (2**a)
        
        # Find max k such that product of k smallest odd primes <= remaining
        prod = 1
        k = 0
        for p in _SMALL_PRIMES[1:]:  # odd primes
            if prod * p > remaining:
                break
            prod *= p
            k += 1
        
        # Try all k' from 0 to k
        for kp in range(0, k + 1):
            if kp == 0:
                n = 2**a
            else:
                pprod = 1
                for i in range(kp):
                    pprod *= _SMALL_PRIMES[i + 1]
                n = (2**a) * pprod
            
            if n > LIMIT:
                break
            
            s = _score_bruteforce(n)
            consider(n, s)
            
            if time.time() > DEADLINE - 10:
                return best_n
    
    # Phase 2: Try more general forms with different prime sets
    # Use greedy: for fixed a, choose primes to maximize score
    for a in range(1, 41):
        if time.time() > DEADLINE - 20:
            return best_n
        base = 2**a
        remaining = LIMIT // base
        
        # Try different combinations of odd primes
        # Start with smallest, then try swapping in larger ones
        for k in range(1, 20):
            if time.time() > DEADLINE - 10:
                return best_n
            
            # Base case: k smallest odd primes
            if k > len(_SMALL_PRIMES) - 1:
                break
            
            prod = 1
            for i in range(k):
                prod *= _SMALL_PRIMES[i + 1]
            
            if prod > remaining:
                break
            
            n = base * prod
            s = _score_bruteforce(n)
            consider(n, s)
            
            # Try swapping the largest prime for a larger one
            if k >= 1:
                largest_idx = k  # index in _SMALL_PRIMES (0-based, so k-th odd prime is at index k+1)
                current_largest = _SMALL_PRIMES[k]  # k-th odd prime (0-indexed in odd primes)
                
                # Try replacing with next few primes
                for replacement in _SMALL_PRIMES[k+1:k+5]:
                    if replacement * prod // current_largest > remaining:
                        break
                    new_prod = prod // current_largest * replacement
                    n = base * new_prod
                    if n <= LIMIT:
                        s = _score_bruteforce(n)
                        consider(n, s)
    
    return best_n
# EVOLVE-BLOCK-END
