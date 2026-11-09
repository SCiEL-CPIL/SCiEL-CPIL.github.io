import math
import time

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n via trial division."""
    if n == 1:
        return 1
    result = 1
    tmp = n
    p = 2
    while p * p <= tmp:
        if tmp % p == 0:
            term = 1
            power = 1
            while tmp % p == 0:
                tmp //= p
                power *= p
                term += power
            result *= term
        p += 1 if p == 2 else 2
    if tmp > 1:
        result *= (1 + tmp)
    return result

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Base known solution
    best_M = 333312
    best_k = [336, 372, 434]

    # Search strategy:
    # 1. Try M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f (small primes, many divisors)
    #    For each such M, find all k | M with k <= sqrt(M) and k*sigma(k) = M
    # 2. Also try M = P * Q where P, Q are products of small primes

    # Generate candidate M values with many divisors
    # Focus on M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g
    # where each exponent is small (0-4) to keep M <= 10^18

    # Precompute: for a given M, we need k | M, k <= sqrt(M), and sigma(k) = M/k
    # So k must be a divisor of M, and sigma(k) must also be a divisor of M (since M/k is integer)

    # Strategy: iterate over possible k values (up to sqrt(M_max) ~ 10^9 is too large)
    # Better: iterate over M candidates, then factor M, enumerate divisors k, check condition

    # Let's try a different approach:
    # For k up to some limit, compute sigma(k), then M = k * sigma(k)
    # Group by M, count distinct k values
    # But k can be up to 10^9, so we need to be smart

    # Approach: For small k (up to ~50000), compute k*sigma(k) and group
    # For larger k, use the constraint that k | M = k*sigma(k) means sigma(k) | M/k... 
    # Actually k | M is automatic since M = k*sigma(k).
    # The constraint is k <= sqrt(M) = sqrt(k*sigma(k)), i.e., k <= sigma(k), which is always true.
    # So the real constraint is just that we want many k mapping to the same M.

    # So: compute f(k) = k * sigma(k) for many k, find M with most preimages.
    # Limit k to reasonable range. Let's try k up to 200000.

    from collections import defaultdict
    M_to_ks = defaultdict(list)

    # Phase 1: scan k from 1 to K_max
    K_max = 300000
    for k in range(1, K_max + 1):
        if time.time() > deadline - 30:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            # k <= sqrt(M) means k^2 <= M = k*sigma(k) means k <= sigma(k), always true
            M_to_ks[M].append(k)

    # Find best M from this scan
    for M, ks in M_to_ks.items():
        if len(ks) > len(best_k):
            best_M = M
            best_k = ks

    # Phase 2: Try structured M values
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # For each such M, enumerate divisors k of M with k <= sqrt(M) and check k*sigma(k)=M

    def _divisors(n):
        """Generate all divisors of n."""
        divs = [1]
        tmp = n
        p = 2
        while p * p <= tmp:
            if tmp % p == 0:
                power = 1
                new_divs = []
                while tmp % p == 0:
                    tmp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs.extend(new_divs)
            p += 1 if p == 2 else 2
        if tmp > 1:
            for d in list(divs):
                divs.append(d * tmp)
        return divs

    # Try M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i
    # with small exponents
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Generate M candidates with many divisors
    # M = product of p_i^e_i where e_i in {0,1,2,3}
    # Focus on combinations that give M <= 10^18
    
    # Try specific structured M values known to have many solutions
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g
    # Let's try a grid search over exponents
    
    def _try_M(M):
        """Check a specific M for solutions."""
        if M < 1 or M > 10**18:
            return []
        sqrt_M = math.isqrt(M)
        divs = _divisors(M)
        ks = []
        for k in divs:
            if k > sqrt_M:
                break
            if k * _sigma(k) == M:
                ks.append(k)
        return ks

    # Try M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h
    # with exponents 0-4, but keep M <= 10^18
    # This is a large space, so be selective
    
    # Try M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # a in 0..10, b in 0..6, c in 0..4, d in 0..3, e in 0..2, f in 0..2
    # That's 11*7*5*4*3*3 = 13860 candidates
    
    # But computing divisors and sigma for each is expensive.
    # Let's be more targeted.
    
    # Try M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g
    # with a+b+c+d+e+f+g <= 20 or so, and M <= 10^18
    
    # Actually, let's try a more focused search:
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # a in 0..8, b in 0..5, c in 0..3, d in 0..2, e in 0..2, f in 0..2
    # = 9*6*4*3*3*3 = 5832 candidates
    
    for a in range(0, 9):
        if time.time() > deadline - 20:
            break
        for b in range(0, 6):
            for c in range(0, 4):
                for d in range(0, 3):
                    for e in range(0, 3):
                        for f in range(0, 3):
                            M = (1 << a) * (3**b) * (5**c) * (7**d) * (11**e) * (13**f)
                            if M > 10**18 or M < 1:
                                continue
                            if time.time() > deadline - 15:
                                break
                            ks = _try_M(M)
                            if len(ks) > len(best_k):
                                best_M = M
                                best_k = ks
                            if len(ks) >= 15:
                                # Found a good one, can stop early
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    # Phase 3: Try M with more prime factors
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h
    for a in range(0, 7):
        if time.time() > deadline - 10:
            break
        for b in range(0, 5):
            for c in range(0, 4):
                for d in range(0, 3):
                    for e in range(0, 3):
                        for f in range(0, 3):
                            for g in range(0, 2):
                                for h in range(0, 2):
                                    M = (1 << a) * (3**b) * (5**c) * (7**d) * (11**e) * (13**f) * (17**g) * (19**h)
                                    if M > 10**18 or M < 1:
                                        continue
                                    if time.time() > deadline - 10:
                                        break
                                    ks = _try_M(M)
                                    if len(ks) > len(best_k):
                                        best_M = M
                                        best_k = ks
                                    if len(ks) >= 20:
                                        break
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    # Ensure best_k is sorted and has valid types
    best_k = sorted(set(int(k) for k in best_k))
    best_M = int(best_M)

    # Verify: all k must satisfy k <= isqrt(M) and k*sigma(k) = M
    verified = []
    sqrt_M = math.isqrt(best_M)
    for k in best_k:
        if k <= sqrt_M and k * _sigma(k) == best_M:
            verified.append(k)
    best_k = verified

    return {'M': best_M, 'k': best_k}
