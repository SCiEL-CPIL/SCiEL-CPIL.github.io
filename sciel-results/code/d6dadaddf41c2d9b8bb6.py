import time
from math import isqrt
from functools import lru_cache

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Compute square-free parts using smallest prime factor
    max_n = 5000
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_sf(n):
        if n <= 1:
            return 1
        result = 1
        x = n
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt ^= 1
            if cnt:
                result *= p
        return result
    
    # Precompute sf for all numbers in range
    sf_arr = [0] * (max_n + 1)
    sf_arr[1] = 1
    for i in range(2, max_n + 1):
        sf_arr[i] = get_sf(i)
    
    # Target: sf(210) = 210 (since 210 = 2*3*5*7, all to first power)
    target_sf = 210
    
    # Collect all primes up to max_n
    primes = []
    for i in range(2, max_n + 1):
        if spf[i] == i:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def sf_to_bits(sf_val):
        """Convert square-free integer to bit vector (as Python int bitmask)."""
        bits = 0
        x = sf_val
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                bits |= (1 << prime_to_idx[p])
                x //= p
        if x > 1:
            if x in prime_to_idx:
                bits |= (1 << prime_to_idx[x])
        return bits
    
    target_bits = sf_to_bits(target_sf)
    
    def find_subset_gauss(bits_list, target):
        """
        Find a subset of bits_list whose XOR equals target.
        Returns list of indices or None.
        Uses Gaussian elimination over GF(2) with index tracking.
        """
        n = len(bits_list)
        if n == 0:
            return None
        
        if target == 0:
            return None  # We need non-empty subset; zero target means dependent set
        
        # basis: dict from pivot_bit_position -> (reduced_vector, frozenset of original indices)
        basis = {}
        
        for i, v in enumerate(bits_list):
            if v == 0:
                continue
            cur = v
            used = frozenset({i})
            hb = cur.bit_length() - 1
            while hb >= 0 and hb in basis:
                bv, bused = basis[hb]
                cur ^= bv
                used = used.symmetric_difference(bused)
                if cur == 0:
                    break
                hb = cur.bit_length() - 1
            if cur == 0:
                continue
            hb = cur.bit_length() - 1
            basis[hb] = (cur, used)
        
        # Try to express target using the basis
        cur = target
        used = frozenset()
        hb = cur.bit_length() - 1
        while hb >= 0 and hb in basis:
            bv, bused = basis[hb]
            cur ^= bv
            used = used.symmetric_difference(bused)
            if cur == 0:
                break
            hb = cur.bit_length() - 1
        
        if cur != 0:
            return None
        
        if not used:
            return None
        
        return sorted(used)
    
    def check_M(M):
        """Check if there's a valid subset with all elements in (210, M]."""
        if M < 211:
            return None
        bits_list = [sf_to_bits(sf_arr[n]) for n in range(211, M + 1)]
        return find_subset_gauss(bits_list, target_bits)
    
    best_solution = None
    best_M = max_n + 1
    
    # Binary search for minimal M
    lo = 211
    hi = max_n
    
    # First verify hi works
    result_hi = check_M(hi)
    if result_hi is None:
        # Fallback: try known solution
        best_solution = [224, 240]
        return [int(x) for x in best_solution]
    
    while lo < hi and time.time() < deadline - 5.0:
        mid = (lo + hi) // 2
        result_mid = check_M(mid)
        if result_mid is not None:
            best_M = mid
            hi = mid
        else:
            lo = mid + 1
    
    # lo == hi, verify
    if time.time() < deadline - 5.0:
        result_lo = check_M(lo)
        if result_lo is not None:
            best_M = lo
    
    if best_M <= max_n:
        idxs = check_M(best_M)
        if idxs is not None:
            selected = [211 + i for i in idxs]
            prod = 210
            for x in selected:
                prod *= x
            if _is_perfect_square(prod) and len(selected) >= 1 and len(selected) <= 100:
                best_solution = selected
    
    if best_solution is None:
        # Try incremental search from 211 upward for small M
        for M in range(211, min(600, max_n + 1)):
            if time.time() > deadline - 5.0:
                break
            idxs = check_M(M)
            if idxs is not None:
                selected = [211 + i for i in idxs]
                prod = 210
                for x in selected:
                    prod *= x
                if _is_perfect_square(prod) and len(selected) >= 1 and len(selected) <= 100:
                    best_solution = selected
                    break
    
    if best_solution is None:
        best_solution = [224, 240]
    
    # Verify all constraints
    for x in best_solution:
        if not isinstance(x, int) or isinstance(x, bool):
            best_solution = [224, 240]
            break
        if x <= 210 or x > 10210:
            best_solution = [224, 240]
            break
    if len(best_solution) < 1 or len(best_solution) > 100:
        best_solution = [224, 240]
    if len(set(best_solution)) != len(best_solution):
        best_solution = [224, 240]
    
    prod = 210
    for x in best_solution:
        prod *= x
    if not _is_perfect_square(prod):
        best_solution = [224, 240]
    
    return [int(x) for x in best_solution]
