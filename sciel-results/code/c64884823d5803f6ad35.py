"""Combined program: structural parity-map search from inspiration + robust verification from current."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = P['n']  # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']  # 100

    # 210 = 2 * 3 * 5 * 7 (square-free)
    # Target: product of selected numbers * 210 must be a perfect square.
    # This means the parity vector of the product of selected numbers must equal
    # the parity vector of 210 over all primes (i.e., odd exponent for 2,3,5,7; even for all else).

    # Sieve for primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i * i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Target parity set: primes 2, 3, 5, 7 have odd exponent in 210
    target_set = frozenset(prime_to_idx[p] for p in (2, 3, 5, 7) if p in prime_to_idx)

    def get_parity(x):
        """Return frozenset of prime indices with odd exponent in x."""
        odd_primes = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    odd_primes.add(prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                odd_primes.add(prime_to_idx[temp])
            else:
                odd_primes.add(-1)  # shouldn't happen
        return frozenset(odd_primes)

    # Precompute parities for all candidate numbers
    parities = {}
    for x in range(211, max_val + 1):
        parities[x] = get_parity(x)

    # Build parity map: parity -> sorted list of numbers with that parity
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)

    def verify(solution):
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) < 1 or len(solution) > max_size:
            return False
        if len(set(solution)) != len(solution):
            return False
        for x in solution:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x <= n or x > max_val:
                return False
        prod = n
        for x in solution:
            prod *= x
        r = math.isqrt(prod)
        if r * r != prod:
            return False
        return True

    # Start with known solution: [840] gives 840*210 = 176400 = 420^2
    best_solution = [840]
    best_raw = 630

    # Check size 1: x = 210 * k^2, x > 210, x <= max_val
    for k in range(1, 20):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            if _is_square(x * 210):
                raw = x - 210
                if raw < best_raw:
                    best_raw = raw
                    best_solution = [x]

    # Check size 2: for each x in (210, best_max), find y such that parity(x) XOR parity(y) = target_set
    current_best_max = best_raw + 210

    for x in range(211, current_best_max):
        if time.time() > deadline:
            break
        px = parities[x]
        py_needed = px ^ target_set
        if py_needed in parity_map:
            for y in parity_map[py_needed]:
                if y == x or y <= 210:
                    continue
                max_xy = max(x, y)
                if max_xy < current_best_max:
                    prod = x * y * 210
                    if _is_square(prod):
                        current_best_max = max_xy
                        best_raw = max_xy - 210
                        best_solution = [x, y]
                        break
                else:
                    break

    # Check size 3: fix x, y from small range, find z with needed parity
    if best_raw > 50 and time.time() < deadline - 5:
        small_range = list(range(211, min(600, current_best_max)))
        for i in range(len(small_range)):
            if time.time() > deadline:
                break
            x = small_range[i]
            for j in range(i + 1, len(small_range)):
                if time.time() > deadline:
                    break
                y = small_range[j]
                needed = target_set ^ parities[x] ^ parities[y]
                if needed in parity_map:
                    for z in parity_map[needed]:
                        if z == x or z == y or z <= 210:
                            continue
                        max_xyz = max(x, y, z)
                        if max_xyz < current_best_max:
                            prod = x * y * z * 210
                            if _is_square(prod):
                                current_best_max = max_xyz
                                best_raw = max_xyz - 210
                                best_solution = [x, y, z]
                                break
                        else:
                            break

    # Final verification
    if not verify(best_solution):
        best_solution = [840]
        best_raw = 630

    return list(best_solution)
# EVOLVE-BLOCK-END
