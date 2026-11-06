"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd, isqrt

DEADLINE = 140.0  # safety margin below 150s evaluator limit

def _sieve_primes(limit):
    """Return list of all primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _miller_rabin(n, a_list):
    """Deterministic Miller-Rabin for n < 3.3e24 using given witnesses."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in a_list:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _is_prime(n):
    """Fast primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    return _miller_rabin(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])

def _pollard_brent(n, rand_func=None):
    """Pollard's rho with Brent's improvement."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    if rand_func is None:
        rand_func = random.randrange
    y = rand_func(1, n - 1)
    c = rand_func(1, n - 1)
    m = 128
    g = r = q = 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = q * abs(x - y) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    if g == n:
        return None
    return g

def _check_run(p, small_primes, max_k=200, deadline=None):
    """
    Check how many consecutive integers p^2+1, ..., p^2+k have all prime factors <= p.
    Fast segmented sieve + Pollard-Brent for remainders.
    """
    if deadline is not None and time.time() > deadline:
        return 0
    p2 = p * p
    max_k = min(max_k, 200)
    
    # remaining[i] holds (p^2 + i) with small prime factors removed
    remaining = [0] * (max_k + 1)
    for i in range(1, max_k + 1):
        remaining[i] = p2 + i
    
    # Sieve: for each prime q <= p, divide out q from remaining[i] where applicable
    for q in small_primes:
        if q > p:
            break
        if deadline is not None and time.time() > deadline:
            break
        # First multiple of q in [p2+1, p2+max_k]
        start = ((p2 + 1 + q - 1) // q) * q
        for val in range(start, p2 + max_k + 1, q):
            idx = val - p2
            if 1 <= idx <= max_k:
                while remaining[idx] % q == 0:
                    remaining[idx] //= q
    
    # Now walk from i=1 to find the longest run where remaining[i] is p-smooth
    k = 0
    for i in range(1, max_k + 1):
        rem = remaining[i]
        if rem == 1:
            k += 1
            continue
        # rem > 1: check if all prime factors of rem are <= p
        if rem > p:
            # Could still be p-smooth if rem is composite with factors <= p
            # But if rem is prime and > p, fail
            if _is_prime(rem):
                break
            # rem is composite; factor it
            is_ok = True
            stack = [rem]
            while stack:
                m = stack.pop()
                if m == 1:
                    continue
                if _is_prime(m):
                    if m > p:
                        is_ok = False
                        break
                    continue
                d = None
                for attempt in range(15):
                    d = _pollard_brent(m)
                    if d is not None and 1 < d < m:
                        break
                if d is None or d == m:
                    if m > p:
                        is_ok = False
                        break
                else:
                    if d > p or m // d > p:
                        is_ok = False
                        break
                    stack.append(d)
                    stack.append(m // d)
                if deadline is not None and time.time() > deadline:
                    return k
            if not is_ok:
                break
        else:
            # rem <= p, so all factors are <= p (trivially)
            pass
        k += 1
        if deadline is not None and time.time() > deadline:
            break
    return k

def _crt_combine(constraints):
    """
    constraints: list of (residue, modulus) pairs.
    Returns (solution, lcm) or (None, None) if inconsistent.
    Uses iterative CRT.
    """
    if not constraints:
        return (0, 1)
    result = constraints[0][0]
    mod = constraints[0][1]
    for (r, m) in constraints[1:]:
        # Solve x ≡ result (mod mod), x ≡ r (mod m)
        # x = result + t*mod
        # result + t*mod ≡ r (mod m)
        # t*mod ≡ r - result (mod m)
        g = gcd(mod, m)
        diff = r - result
        if diff % g != 0:
            return (None, None)
        # t*mod/g ≡ diff/g (mod m/g)
        mod_g = mod // g
        m_g = m // g
        diff_g = diff // g
        # Find inverse of mod_g mod m_g
        try:
            inv = pow(mod_g, -1, m_g)
        except:
            return (None, None)
        t = (diff_g * inv) % m_g
        new_mod = mod * m_g
        result = result + t * mod
        result %= new_mod
        mod = new_mod
    return (result, mod)

def _quadratic_residues(a, p):
    """Find all x in [0, p-1] such that x^2 ≡ a (mod p)."""
    a %= p
    sols = []
    for x in range(p):
        if (x * x) % p == a:
            sols.append(x)
    return sols

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0
    
    # Precompute small primes
    small_primes = _sieve_primes(1_000_000)
    
    best_k = 0
    best_p = 7
    
    # Phase 1: Quick check of small primes
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        k = _check_run(p, small_primes, max_k=200, deadline=deadline)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: CRT-based candidate generation
    # Strategy: pick a set of small primes Q = {q1, q2, ..., qm} and offsets i1, i2, ..., im
    # such that p^2 + ij ≡ 0 (mod qj) for each j.
    # This means p ≡ ±sqrt(-ij) (mod qj).
    # By CRT, we get p ≡ r (mod M) where M = product of qj.
    # Then p = r + t*M. We need p to be prime and <= 10^9.
    
    # Use small primes for CRT constraints
    crt_primes_pool = [q for q in small_primes if 3 <= q <= 200]
    
    rng = random.Random(42)
    candidates_checked = set()
    
    num_crt_rounds = 5000
    for round_idx in range(num_crt_rounds):
        if time.time() > deadline - 5:
            break
        
        # Choose 2-4 small primes for CRT
        num_constraints = rng.randint(2, 4)
        chosen_q = rng.sample(crt_primes_pool, min(num_constraints, len(crt_primes_pool)))
        
        constraints = []
        consistent = True
        for q in chosen_q:
            # Pick a random offset i in [1, 20]
            i = rng.randint(1, 20)
            # Solve p^2 ≡ -i (mod q)
            a = (-i) % q
            sols = _quadratic_residues(a, q)
            if not sols:
                consistent = False
                break
            # Pick one solution randomly
            r = rng.choice(sols)
            constraints.append((r, q))
        
        if not consistent:
            continue
        
        sol, M = _crt_combine(constraints)
        if sol is None or M <= 0:
            continue
        
        # M is the product of chosen_q, so M is at most 200^4 = 1.6e9
        # p = sol + t*M, need p <= 10^9
        if M > 10**9:
            continue
        
        # Generate a few candidates: t = 0, 1, 2, ...
        # p must be prime
        num_t = min(20, (10**9 - sol) // M + 1) if sol <= 10**9 else 0
        if num_t <= 0:
            # Try sol + M, sol + 2M, etc.
            t_start = max(1, (100 - sol + M - 1) // M)
            t_end = min(t_start + 20, (10**9 - sol) // M)
            t_range = range(t_start, t_end + 1)
        else:
            t_range = range(0, num_t)
        
        for t in t_range:
            p = sol + t * M
            if p < 11 or p > 10**9:
                continue
            if p in candidates_checked:
                continue
            candidates_checked.add(p)
            
            if not _is_prime(p):
                continue
            
            k = _check_run(p, small_primes, max_k=200, deadline=deadline)
            if k > best_k:
                best_k = k
                best_p = p
                # Try nearby values
                if best_k >= 3 and time.time() < deadline - 10:
                    for delta in range(-50, 51):
                        if delta == 0:
                            continue
                        np_ = p + delta
                        if 11 <= np_ <= 10**9 and _is_prime(np_) and np_ not in candidates_checked:
                            candidates_checked.add(np_)
                            k2 = _check_run(np_, small_primes, max_k=200, deadline=deadline)
                            if k2 > best_k:
                                best_k = k2
                                best_p = np_
                        if time.time() > deadline:
                            break
        
        if time.time() > deadline:
            break
    
    # Phase 3: Random sampling backup
    if time.time() < deadline - 5:
        rng2 = random.Random(77777)
        for _ in range(500):
            if time.time() > deadline:
                break
            x = rng2.randint(10**4, 10**9)
            p = None
            for d in range(0, 1000):
                for cand in [x + d, x - d]:
                    if 11 <= cand <= 10**9 and cand not in candidates_checked and _is_prime(cand):
                        p = cand
                        break
                if p:
                    break
            if p is None:
                continue
            candidates_checked.add(p)
            k = _check_run(p, small_primes, max_k=200, deadline=deadline)
            if k > best_k:
                best_k = k
                best_p = p
    
    if best_k < 1:
        best_p = 7
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
