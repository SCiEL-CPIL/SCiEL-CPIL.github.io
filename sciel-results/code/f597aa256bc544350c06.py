import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    LIMIT = 10**18
    LOG_LIMIT = 60.0  # log2(10^18) ≈ 59.79

    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors

    def tau_perp(n, factors=None):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if factors is None:
            factors = []
            temp = n
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
        
        divisors = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, len(divisors)

    best_n = 1
    best_score = 0

    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT:
            return -1
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        return score

    # Baseline: product of first 15 primes (square-free primorial)
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    if n_base <= LIMIT:
        score_base = try_candidate([(p, 1) for p in primes_15])

    if time.time() - start_time > DEADLINE:
        return best_n

    # === STRUCTURAL CHANGE: Search over "small-prime-heavy" exponent vectors ===
    # Model: n = 2^a * 3^b * 5^c * 7^d * (square-free tail of primes)
    # The log2 budget is log2(n) = a*log2(2) + b*log2(3) + c*log2(5) + d*log2(7) + sum(log2(p_i))
    # We use a greedy/DP approach: allocate budget to small primes first (they create
    # the most divisors per unit log-budget), then fill remaining with square-free primes.
    
    # Key insight from packing model: increasing exponent of small prime p adds (e+1)
    # "layers" of divisors. Adjacent coprime pairs occur at boundaries where consecutive
    # divisors in sorted order have disjoint prime supports. Small primes with high
    # exponents create dense interleaving that can maximize these boundaries.
    
    # Strategy: enumerate exponent vectors (a,b,c,d) for primes 2,3,5,7 with
    # a*log2(2) + b*log2(3) + c*log2(5) + d*log2(7) <= LOG_LIMIT * 0.7 (reserve for tail)
    # Then greedily add square-free primes from 11 upward to fill remaining budget.
    
    log2_2 = math.log2(2)
    log2_3 = math.log2(3)
    log2_5 = math.log2(5)
    log2_7 = math.log2(7)
    
    # Reserve fraction of budget for tail primes
    reserve_frac = 0.35
    
    max_a = int(LOG_LIMIT * (1 - reserve_frac) / log2_2)
    max_b = int(LOG_LIMIT * (1 - reserve_frac) / log2_3)
    max_c = int(LOG_LIMIT * (1 - reserve_frac) / log2_5)
    max_d = int(LOG_LIMIT * (1 - reserve_frac) / log2_7)
    
    # Limit search space for feasibility
    max_a = min(max_a, 12)
    max_b = min(max_b, 8)
    max_c = min(max_c, 6)
    max_d = min(max_d, 5)
    
    tail_primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    evaluated = 0
    
    for a in range(max_a + 1):
        if time.time() - start_time > DEADLINE:
            break
        for b in range(max_b + 1):
            if time.time() - start_time > DEADLINE:
                break
            log_ab = a * log2_2 + b * log2_3
            if log_ab > LOG_LIMIT * (1 - reserve_frac):
                break
            for c in range(max_c + 1):
                if time.time() - start_time > DEADLINE:
                    break
                log_abc = log_ab + c * log2_5
                if log_abc > LOG_LIMIT * (1 - reserve_frac):
                    break
                for d in range(max_d + 1):
                    if time.time() - start_time > DEADLINE:
                        break
                    log_abcd = log_abc + d * log2_7
                    if log_abcd > LOG_LIMIT * (1 - reserve_frac):
                        break
                    
                    # Build base factors
                    factors = []
                    if a > 0:
                        factors.append((2, a))
                    if b > 0:
                        factors.append((3, b))
                    if c > 0:
                        factors.append((5, c))
                    if d > 0:
                        factors.append((7, d))
                    
                    # Greedily add tail primes (square-free) to fill remaining budget
                    remaining_log = LOG_LIMIT - log_abcd
                    for p in tail_primes:
                        if math.log2(p) > remaining_log:
                            break
                        factors.append((p, 1))
                        remaining_log -= math.log2(p)
                    
                    # Evaluate this candidate
                    n = 1
                    for p, e in factors:
                        n *= p ** e
                    if n > LIMIT:
                        # Try removing the last added prime
                        if len(factors) > 4:
                            factors.pop()
                            n //= factors[-1][0] if factors[-1][1] == 1 else factors[-1][0] ** factors[-1][1]
                        else:
                            continue
                        if n > LIMIT:
                            continue
                    
                    score = try_candidate(factors)
                    evaluated += 1
                    
                    if evaluated % 50 == 0 and time.time() - start_time > DEADLINE:
                        break

    if time.time() - start_time > DEADLINE:
        return best_n

    # === REFINEMENT: Local search around best found ===
    # Try swapping tail primes for slightly larger ones, or adjusting exponents by +/-1
    if best_n > 1:
        # Factorize best_n
        def factorize(n):
            factors = []
            temp = n
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
            return factors
        
        best_factors = factorize(best_n)
        
        # Try increasing each exponent by 1 (if within limit)
        for i, (p, e) in enumerate(best_factors):
            if time.time() - start_time > DEADLINE:
                break
            new_factors = list(best_factors)
            new_factors[i] = (p, e + 1)
            n_test = 1
            for pp, ee in new_factors:
                n_test *= pp ** ee
            if n_test <= LIMIT:
                try_candidate(new_factors)
        
        # Try decreasing each exponent by 1 and adding a new prime
        for i, (p, e) in enumerate(best_factors):
            if time.time() - start_time > DEADLINE:
                break
            if e > 1:
                new_factors = list(best_factors)
                new_factors[i] = (p, e - 1)
                n_test = 1
                for pp, ee in new_factors:
                    n_test *= pp ** ee
                # Find next available prime
                max_p = max(pp for pp, _ in new_factors)
                for cand in range(max_p + 1, max_p + 50):
                    # Check if prime
                    if cand < 2:
                        continue
                    is_p = True
                    for d in range(2, int(math.isqrt(cand)) + 1):
                        if cand % d == 0:
                            is_p = False
                            break
                    if not is_p:
                        continue
                    # Check if already in factors
                    if any(pp == cand for pp, _ in new_factors):
                        continue
                    new_factors.append((cand, 1))
                    n_test2 = 1
                    for pp, ee in new_factors:
                        n_test2 *= pp ** ee
                    if n_test2 <= LIMIT:
                        try_candidate(new_factors)
                    else:
                        break

    return best_n
