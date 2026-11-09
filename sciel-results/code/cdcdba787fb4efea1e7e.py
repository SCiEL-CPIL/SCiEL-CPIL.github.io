import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # Sieve for smallest prime factor for fast factorization
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Collect all primes up to max_val
    primes = [i for i in range(2, max_val + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Target: primes 2,3,5,7 have odd parity, all others even.
    target_primes = [2, 3, 5, 7]
    target_set = frozenset(prime_to_idx[p] for p in target_primes if p in prime_to_idx)
    
    def get_parity(x):
        """Return frozenset of prime indices with odd exponent in x."""
        odd_primes = set()
        temp = x
        while temp > 1:
            p = spf[temp]
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            if cnt % 2 == 1:
                if p in prime_to_idx:
                    odd_primes.add(prime_to_idx[p])
                else:
                    odd_primes.add(-1)
        return frozenset(odd_primes)
    
    # Precompute parities for numbers 211 to 10210
    parities = {}
    for x in range(211, max_val + 1):
        parities[x] = get_parity(x)
    
    # Build a map from parity to sorted list of numbers
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)
    
    best_result = [840]  # fallback
    best_raw = 630
    current_best_max = 840
    
    # Search for size 1
    for k in range(1, 10):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            if _is_square(x * 210):
                raw = x - 210
                if raw < best_raw:
                    best_raw = raw
                    current_best_max = x
                    best_result = [x]
    
    # Search for size 2
    for x in range(211, current_best_max):
        if time.time() > deadline:
            break
        px = parities[x]
        py = px ^ target_set
        if py in parity_map:
            for y in parity_map[py]:
                if y <= 210:
                    continue
                if y == x:
                    continue
                max_xy = max(x, y)
                if max_xy >= current_best_max:
                    break
                prod = x * y * 210
                if _is_square(prod):
                    current_best_max = max_xy
                    best_raw = max_xy - 210
                    best_result = [x, y]
                    break
    
    # Search for size 3 with optimized approach
    if best_raw > 10 and time.time() < deadline:
        upper_bound = min(current_best_max, 500)
        small_numbers = list(range(211, upper_bound))
        n_small = len(small_numbers)
        for i in range(n_small):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            for j in range(i + 1, n_small):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                if max(x, y) >= current_best_max:
                    break
                needed_parity = target_set ^ parities[x] ^ parities[y]
                if needed_parity in parity_map:
                    found = False
                    for z in parity_map[needed_parity]:
                        if z <= 210 or z == x or z == y:
                            continue
                        max_xyz = max(x, y, z)
                        if max_xyz >= current_best_max:
                            break
                        prod = x * y * z * 210
                        if _is_square(prod):
                            current_best_max = max_xyz
                            best_raw = max_xyz - 210
                            best_result = [x, y, z]
                            found = True
                            break
                    if found:
                        break
            if best_raw <= 10:
                break
    
    # Search for size 4 with very small numbers
    if best_raw > 5 and time.time() < deadline:
        upper4 = min(current_best_max, 350)
        nums4 = list(range(211, upper4))
        n4 = len(nums4)
        for i in range(n4):
            if time.time() > deadline:
                break
            x = nums4[i]
            for j in range(i + 1, n4):
                if time.time() > deadline:
                    break
                y = nums4[j]
                if max(x, y) >= current_best_max:
                    break
                for k in range(j + 1, n4):
                    if time.time() > deadline:
                        break
                    z = nums4[k]
                    if max(x, y, z) >= current_best_max:
                        break
                    needed_parity = target_set ^ parities[x] ^ parities[y] ^ parities[z]
                    if needed_parity in parity_map:
                        found = False
                        for w in parity_map[needed_parity]:
                            if w <= 210 or w in (x, y, z):
                                continue
                            max_xyzw = max(x, y, z, w)
                            if max_xyzw >= current_best_max:
                                break
                            prod = x * y * z * w * 210
                            if _is_square(prod):
                                current_best_max = max_xyzw
                                best_raw = max_xyzw - 210
                                best_result = [x, y, z, w]
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
            if found:
                break
    
    # Return as list of Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
