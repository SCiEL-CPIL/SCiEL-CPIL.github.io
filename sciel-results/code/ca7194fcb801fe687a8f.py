"""Combined program: NumPy sieve enumeration + smooth-number M generation + structural construction."""
import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) by trial division up to sqrt(n)."""
    if n <= 1:
        return 1
    result = 1
    temp = n
    cnt = 0
    while temp % 2 == 0:
        temp //= 2
        cnt += 1
    if cnt:
        result *= (2**(cnt+1) - 1)
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            result *= (p**(cnt+1) - 1) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _verify_solution(M, ks):
    """Verify all k in list are valid distinct solutions for M."""
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        if k in seen:
            return False
        seen.add(k)
        if k < 1 or k * k > M:
            return False
        if k * _sigma(k) != M:
            return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0  # safety margin before 150s kill

    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    # M_to_ks: global accumulator of candidate solutions
    M_to_ks = defaultdict(list)

    def update_best():
        nonlocal best_result, best_count
        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                if _verify_solution(M, ks):
                    best_count = len(ks)
                    best_result = {'M': int(M), 'k': [int(x) for x in ks[:1000]]}

    # ---- Phase 1: NumPy sieve enumeration of k up to N_max ----
    # This is the fast core: compute sigma via sieve, then group by M.
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        N_max = 5000000
        sigma = np.zeros(N_max + 1, dtype=np.int64)
        for i in range(1, N_max + 1):
            sigma[i::i] += i

        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_result

        k_arr = np.arange(1, N_max + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:N_max + 1]
        # k <= sqrt(M)  <=>  k^2 <= M  <=>  k <= sigma(k)
        valid = (M_arr <= 10**18) & (k_arr <= sigma[1:N_max + 1])

        for idx in range(N_max):
            if valid[idx]:
                k = int(k_arr[idx])
                M = int(M_arr[idx])
                M_to_ks[M].append(k)

        update_best()
    else:
        # Fallback: pure Python sieve
        N_max = 1000000
        sigma = [0] * (N_max + 1)
        for i in range(1, N_max + 1):
            for j in range(i, N_max + 1, i):
                sigma[j] += i
        for k in range(1, N_max + 1):
            M = k * sigma[k]
            if M <= 10**18 and k <= sigma[k]:
                M_to_ks[M].append(k)
        update_best()

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_result

    # ---- Phase 2: Smooth-number generation ----
    # Generate k values from restricted small primes with bounded exponents.
    # These produce M = k*sigma(k) values; collisions reveal M with many solutions.
    # This explores k values beyond the sieve range and captures structural collisions.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    # Max k value from smooth generation
    k_max_smooth = 200000000  # 2e8

    def _generate_smooth_ks(primes, max_k):
        """Generate all k <= max_k composed only of primes in primes list."""
        ks = set()
        def _gen(idx, current):
            if current > max_k:
                return
            ks.add(current)
            if idx >= len(primes):
                return
            p = primes[idx]
            val = current
            while val <= max_k:
                _gen(idx + 1, val)
                val *= p
                if val > max_k:
                    break
        _gen(0, 1)
        return ks

    # Use a smaller prime set for manageable generation
    smooth_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    smooth_ks = _generate_smooth_ks(smooth_primes, k_max_smooth)

    # For each smooth k, compute M = k * sigma(k) and add to M_to_ks
    # Skip k values already covered by the sieve (k <= N_max)
    sieve_limit = N_max if 'N_max' in dir() else 1000000
    for k in smooth_ks:
        if k <= sieve_limit:
            continue  # already covered
        if k * k > 10**18:
            continue  # k > sqrt(10^18) = 10^9
        # Quick check: k must be <= sqrt(M) => k <= sigma(k)
        # sigma(k) >= k+1 always (for k >= 1), so this is always true for k >= 1
        # But we need k*sigma(k) <= 10^18
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            M_to_ks[M].append(k)

        if time.time() - start_time > DEADLINE:
            break

    update_best()

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_result

    # ---- Phase 3: Targeted structural construction ----
    # Look for M values with many solutions by analyzing the structure.
    # If M = k1*sigma(k1) = k2*sigma(k2) = ... = kn*sigma(kn),
    # we want to find M that maximizes n.
    # 
    # Strategy: For the top candidate M values (by current solution count),
    # try to find additional solutions by searching divisors of M.
    # k must divide M (since k*sigma(k)=M and k|sigma(k)*k=M... actually k need not divide M).
    # Wait: k*sigma(k) = M does NOT imply k divides M in a useful way for finding k.
    # But k must be a divisor of M only if sigma(k) is an integer (which it is).
    # Actually k*sigma(k) = M means k divides M and sigma(k) = M/k.
    # So k must be a divisor of M!
    
    # For top M candidates, factor M and enumerate divisors to find all valid k.
    top_candidates = sorted(M_to_ks.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    
    for M, existing_ks in top_candidates:
        if time.time() - start_time > DEADLINE:
            break
        sqrt_M = math.isqrt(M)
        # Find all divisors of M up to sqrt_M
        # Factor M first
        temp = M
        factors = {}
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt += 1
                factors[d] = cnt
            d += 1 if d == 2 else 2
        if temp > 1:
            factors[temp] = 1
        
        # Generate all divisors
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        
        # Filter divisors k where k <= sqrt_M and k*sigma(k) == M
        existing_set = set(existing_ks)
        for k in divisors:
            if k > sqrt_M:
                continue
            if k in existing_set:
                continue
            if k * _sigma(k) == M:
                existing_ks.append(k)
                existing_set.add(k)
        
        if len(existing_ks) > best_count:
            if _verify_solution(M, existing_ks):
                best_count = len(existing_ks)
                best_result = {'M': int(M), 'k': [int(x) for x in existing_ks[:1000]]}

    return best_result
