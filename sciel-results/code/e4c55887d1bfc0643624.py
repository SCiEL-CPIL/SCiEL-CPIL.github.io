"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_powerful(n):
    """Check if n is powerful: every prime factor appears with exponent >= 2. 1 is powerful."""
    if n == 1:
        return True
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            cnt = 0
            while m % d == 0:
                cnt += 1
                m //= d
            if cnt < 2:
                return False
        d += 1 if d == 2 else 2
    if m > 1:
        return False  # remaining factor has exponent 1
    return True

def _generate_gaussian_reps(M, factorization):
    """
    Generate all pairs (x, y) with x^2 + y^2 = M, x >= 0, y >= 0, x <= y.
    Uses unique factorization in Z[i].
    factorization: dict mapping prime p (4k+1) to its exponent in M (must be even, = 2*e)
    Also handles factor 2 (exponent must be even for M to be sum of two squares in the right way)
    and 4k+3 primes (exponents must be even).
    
    For M = 2^(2f) * prod(p_i^(2e_i)) * prod(q_j^(2d_j)) where p_i = 4k+1, q_j = 4k+3:
    The number of representations r_2(M) = 4 * prod(2*e_i + 1) [if all 4k+3 exponents are even]
    
    We generate reps via: for each 4k+1 prime p_i = a_i^2 + b_i^2, we have p_i = (a_i + b_i*i)(a_i - b_i*i)
    M^(1/2) = 2^f * prod(p_i^e_i) * prod(q_j^d_j)
    Each representation corresponds to choosing for each p_i^e_i which conjugate to use.
    """
    # First, get sqrt(M) to bound our search
    sqrt_M = isqrt(M)
    if sqrt_M * sqrt_M != M:
        # M is not a perfect square, but we can still have reps
        pass
    
    # For each 4k+1 prime p with exponent 2e in M, we need p = a^2 + b^2
    # Then p^e = (a+bi)^e * (a-bi)^e, and we can choose how many factors of (a+bi) vs (a-bi)
    # Actually, for M = prod p_i^(2e_i), the number of ways is prod(2e_i + 1)
    # Each way corresponds to a choice of k_i in [0, 2e_i] for each i
    # The Gaussian integer is prod (a_i + b_i*i)^(k_i) * (a_i - b_i*i)^(2e_i - k_i)
    
    # Precompute representations of each 4k+1 prime
    prime_reps = {}
    for p, exp in factorization.items():
        if p == 2:
            continue
        if p % 4 == 3:
            continue
        # p = 4k+1, find a^2 + b^2 = p
        a = 1
        found = False
        while a * a <= p:
            b2 = p - a * a
            b = isqrt(b2)
            if b * b == b2 and a <= b:
                prime_reps[p] = (a, b)
                found = True
                break
            a += 1
        if not found:
            prime_reps[p] = (1, 0)  # shouldn't happen for 4k+1 primes
    
    # Generate all combinations of exponents
    # For each 4k+1 prime p with exponent 2e in M, choose k in [0, 2e]
    # The Gaussian integer z = prod over p of (a_p + b_p*i)^k * (a_p - b_p*i)^(2e-k)
    # Then x = Re(z), y = Im(z), and x^2 + y^2 = M
    
    # To avoid generating too many, note that r_2(M) = 4 * prod(2e_i + 1)
    # We only need x >= 0, y >= 0, x <= y, so divide by 8 roughly
    
    # Build list of (prime, exponent, a, b) for 4k+1 primes
    gaussian_primes = []
    for p, exp in factorization.items():
        if p == 2 or p % 4 == 3:
            continue
        if p in prime_reps:
            a, b = prime_reps[p]
            gaussian_primes.append((p, exp, a, b))
    
    if not gaussian_primes:
        # M = 2^(2f) or M = prod of 4k+3 primes with even exponents
        # Only rep is (sqrt(M), 0) or (0, sqrt(M)) if M is a square
        if sqrt_M * sqrt_M == M:
            return [(0, sqrt_M)] if sqrt_M > 0 else []
        return []
    
    # Use recursive generation of Gaussian integers
    reps_set = set()
    
    def _gen(idx, real, imag):
        if time.time() > 140:
            return
        if idx == len(gaussian_primes):
            x, y = abs(real), abs(imag)
            if x * x + y * y == M:
                if x > y:
                    x, y = y, x
                if x >= 0 and y >= 0:
                    reps_set.add((x, y))
            return
        
        p, exp, a, b = gaussian_primes[idx]
        # For this prime, we choose k in [0, exp] (since total exponent is 2*exp, and we split)
        # Wait, the exponent in M is exp (which should be even). Let's say exp = 2*e.
        # Then we choose k in [0, 2*e] for the number of (a+bi) factors.
        # But that's too many. Actually, by symmetry, choosing k and 2e-k gives conjugates.
        # We just need to enumerate all k in [0, exp] where exp is the actual exponent.
        
        for k in range(exp + 1):
            # Compute (a+bi)^k * (a-bi)^(exp-k)
            # This is messy. Let's use a different approach.
            # 
            # Actually, let's just do brute force for small M since our M values are structured.
            # The key insight: for M = (p1*p2*...*pk)^2, the reps can be generated
            # by multiplying Gaussian integers.
            
            # Let's compute (a+bi)^k * (a-bi)^(exp-k) iteratively
            # Start with 1 + 0i
            rr, ri = 1, 0
            # Multiply by (a+bi) k times
            for _ in range(k):
                nr = rr * a - ri * b
                ni = rr * b + ri * a
                rr, ri = nr, ni
            # Multiply by (a-bi) (exp-k) times
            for _ in range(exp - k):
                nr = rr * a + ri * b
                ni = -rr * b + ri * a
                rr, ri = nr, ni
            
            _gen(idx + 1, real * rr - imag * ri, real * ri + imag * rr)
    
    _gen(0, 1, 0)
    
    return list(reps_set)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Strategy: N = R * M where R is powerful, M is a perfect square of a product of 4k+1 primes
    # Then M = S^2 where S = prod(p_i^e_i)
    # Reps of M: for each (x,y) with x^2+y^2=M, we get a=R*x^2, b=R*y^2
    # a and b are powerful iff R*x^2 and R*y^2 are powerful.
    # If R is powerful and x,y are integers, R*x^2 is powerful iff for every prime p|x, 
    # the exponent of p in R*x^2 is >= 2. Since x^2 contributes even exponents, and R has 
    # exponents >= 2, the sum is >= 2. So a and b are always powerful!
    # 
    # So the problem reduces to: find M = S^2 <= 10^18/R maximizing the number of 
    # representations of M as sum of two squares, with S a product of 4k+1 primes.
    #
    # For M = (p1^e1 * p2^e2 * ... * pk^ek)^2, the number of representations (counting 
    # order and signs) is 4 * prod(2*ei + 1).
    # The number of unordered pairs (x,y) with x>=0, y>=0, x<=y is roughly r_2(M)/8
    # (accounting for symmetries), but we need to be careful about x=0 or x=y cases.
    
    # 4k+1 primes
    primes_4k1 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 629, 641, 653, 661, 673, 677, 701, 709, 717, 733, 757, 761, 769, 773, 797]
    # Filter to actual primes
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
    primes_4k1 = [p for p in primes_4k1 if p % 4 == 1 and _is_prime(p)]
    
    # Try R=1 first (most headroom for M)
    # M = S^2 where S = prod(p_i^e_i), maximize prod(2*e_i+1) subject to S^2 <= 10^18
    # i.e., S <= 10^9
    
    # Greedy: assign exponents to smallest primes first to maximize the product
    # For a given budget of log(S) <= 9*log(10), we want to maximize prod(2*e_i+1)
    # This is like: choose e_i >= 0 to maximize prod(2*e_i+1) subject to sum(e_i * log(p_i)) <= 9*log(10)
    
    # Simple approach: try all subsets of primes with exponent 1, then try adding more exponents
    # S_max = 10^9
    
    S_max = 10**9
    
    best_S = 1
    best_prod = 1
    best_exponents = {}
    
    # Start with S=1, prod=1
    # Try adding primes with exponent 1
    current_S = 1
    current_prod = 1
    current_exp = {}
    
    # Greedy: add primes in order of smallest p first
    for p in primes_4k1:
        if current_S * p > S_max:
            break
        current_S *= p
        current_exp[p] = 1
        current_prod *= 3  # 2*1+1 = 3
    
    # Now try to increase exponents of existing primes
    # For each prime p in current_exp, try increasing its exponent
    improved = True
    while improved:
        improved = False
        for p in list(current_exp.keys()):
            if current_S * p > S_max:
                continue
            # Try increasing exponent of p by 1
            new_S = current_S * p
            new_prod = current_prod * (2 * (current_exp[p] + 1) + 1) // (2 * current_exp[p] + 1)
            if new_S <= S_max and new_prod > best_prod:
                current_S = new_S
                current_exp[p] += 1
                current_prod = new_prod
                best_S = current_S
                best_prod = current_prod
                best_exponents = dict(current_exp)
                improved = True
    
    # Also try different combinations: maybe not using the smallest primes
    # Try all subsets of size k for k = 1, 2, ..., up to what fits
    # This is expensive, so limit to small k
    
    # Better: dynamic programming or just try a few heuristics
    # Heuristic 1: all exponent 1, as many primes as fit
    S1 = 1
    exp1 = {}
    for p in primes_4k1:
        if S1 * p > S_max:
            break
        S1 *= p
        exp1[p] = 1
    prod1 = 3 ** len(exp1)
    
    # Heuristic 2: concentrate on few primes with higher exponents
    # For prime p, exponent e gives factor (2e+1) and costs e*log(p)
    # Best ratio: for small p, higher exponents are better
    # Try: p=5 with exponent e, then p=13, etc.
    
    best_overall_S = max(best_S, S1)
    best_overall_prod = max(best_prod, prod1)
    best_overall_exp = best_exponents if best_prod >= prod1 else exp1
    
    # Try a few more combinations
    # For each subset of first few primes, optimize exponents
    for num_primes in range(1, min(len(primes_4k1), 10) + 1):
        subset = primes_4k1[:num_primes]
        # For this subset, find optimal exponents
        # Greedy: give more exponent to smaller primes
        S_trial = 1
        exp_trial = {}
        prod_trial = 1
        # Start with all exponent 1
        for p in subset:
            if S_trial * p > S_max:
                break
            S_trial *= p
            exp_trial[p] = 1
            prod_trial *= 3
        # Now increase exponents
        changed = True
        while changed:
            changed = False
            for p in sorted(exp_trial.keys()):  # smallest first
                if S_trial * p > S_max:
                    continue
                new_S = S_trial * p
                new_prod = prod_trial * (2 * (exp_trial[p] + 1) + 1) // (2 * exp_trial[p] + 1)
                if new_S <= S_max:
                    S_trial = new_S
                    exp_trial[p] += 1
                    prod_trial = new_prod
                    changed = True
        if prod_trial > best_overall_prod:
            best_overall_prod = prod_trial
            best_overall_S = S_trial
            best_overall_exp = dict(exp_trial)
    
    # Now we have best_overall_S and best_overall_exp
    # M = best_overall_S^2
    M = best_overall_S * best_overall_S
    R = 1
    N = R * M
    
    if N < 2 or N > 10**18:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Generate representations of M as x^2 + y^2
    # Use the Gaussian integer method
    factorization = dict(best_overall_exp)
    reps = _generate_gaussian_reps(M, factorization)
    
    # Filter to x >= 0, y >= 0, x <= y
    pairs = []
    seen = set()
    for x, y in reps:
        if x > y:
            x, y = y, x
        if x >= 0 and y >= 0 and x * x + y * y == M:
            a = R * x * x
            b = R * y * y
            if a + b == N and a <= b and (a, b) not in seen:
                seen.add((a, b))
                pairs.append([a, b])
    
    # Verify
    verified_pairs = []
    for a, b in pairs:
        if a + b == N and a <= b and a >= 1 and b >= 1:
            if _is_powerful(a) and _is_powerful(b):
                verified_pairs.append([a, b])
    
    if len(verified_pairs) > 0:
        return {'N': N, 'pairs': verified_pairs}
    
    # Fallback
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
