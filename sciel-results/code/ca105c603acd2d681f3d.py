import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _factorize(x, primes):
    """Return parity vector (as frozenset of prime indices) for x w.r.t. primes list."""
    parity = []
    temp = x
    for i, p in enumerate(primes):
        if p * p > temp:
            break
        cnt = 0
        while temp % p == 0:
            temp //= p
            cnt ^= 1
        if cnt:
            parity.append(i)
    if temp > 1:
        # temp is a prime not in our list (or a large prime)
        # Find or assign an index
        # We'll handle this by extending primes dynamically
        parity.append(-1)  # marker for "other"
    return tuple(parity)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # Target parity: 210 = 2*3*5*7, all exponent 1 (odd)
    # We need product of selected to have odd parity for 2,3,5,7
    # and even parity for everything else.
    
    # First, let's figure out which primes matter.
    # Numbers up to 10210. Primes up to 10210.
    # But for parity, we only care about primes that appear in factorizations.
    # Let's collect all primes up to max_val.
    
    # Sieve for primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Target: primes 2,3,5,7 have odd parity, all others even.
    # Target vector as a frozenset of indices
    target_primes = [2, 3, 5, 7]
    target_set = frozenset(prime_to_idx[p] for p in target_primes if p in prime_to_idx)
    
    # For each number in (210, 10210], compute its parity vector
    # We'll represent parity as a frozenset of prime indices with odd exponent.
    
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
            # temp is prime
            if temp in prime_to_idx:
                odd_primes.add(prime_to_idx[temp])
            else:
                # Should not happen since we sieved up to max_val
                odd_primes.add(-1)  # shouldn't occur
        return frozenset(odd_primes)
    
    # Build mapping from parity vector to list of numbers with that parity
    parity_to_numbers = defaultdict(list)
    
    # We want to minimize max(selected), so search in increasing order of max value
    # Let's try: for each candidate max M (from 211 upward), find if there's a valid set
    # with all elements <= M.
    
    # Strategy: For small sets (size 1, 2, 3, ...), check if they work.
    # Size 1: need a number x such that x*210 is a perfect square.
    #   x*210 = x * 2 * 3 * 5 * 7 must be a square.
    #   So x must be of form k^2 * 2 * 3 * 5 * 7 / (some square factor)... 
    #   Actually x = 210 * k^2 for some integer k, and x > 210, x <= 10210.
    #   210*k^2 > 210 => k >= 1, 210*k^2 <= 10210 => k^2 <= 48.6 => k <= 6
    #   k=1: x=210 (not > 210)
    #   k=2: x=840, check: 840*210 = 176400, sqrt=420, 420^2=176400. Yes!
    #   So [840] works, max-210 = 630.
    
    # Let's verify: 840 * 210 = 176400. 420^2 = 176400. Correct.
    # So answer 630 is achievable with size 1.
    
    # Can we do better? We need max(selected) < 840.
    # Size 1: only 840 works in range.
    # Size 2: need x*y*210 = square, x,y in (210, M], x != y, M < 840.
    #   x*y must have parity matching target (odd for 2,3,5,7).
    #   So parity(x) XOR parity(y) = target.
    #   i.e., parity(y) = parity(x) XOR target.
    
    # Let's search for size 2 with max < 840.
    
    # Precompute parities for numbers 211 to 10210
    parities = {}
    for x in range(211, max_val + 1):
        parities[x] = get_parity(x)
    
    best_result = [840]  # fallback
    best_raw = 630
    
    # Search for size 1
    for k in range(1, 10):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            if _is_square(x * 210):
                raw = x - 210
                if raw < best_raw:
                    best_raw = raw
                    best_result = [x]
    
    # Search for size 2: for each x in (210, best_max), find y such that parity(x)^parity(y) = target
    # and y in (210, max_val], y != x
    # To minimize max(x,y), we want both small.
    
    # Build a map from parity to sorted list of numbers
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)
    
    # For size 2: iterate x from 211 upward, compute needed parity, check if any y exists
    # with that parity and y > 210, y != x, and max(x,y) < current best_max
    current_best_max = 840  # from size 1
    
    for x in range(211, current_best_max):
        if time.time() > deadline:
            break
        px = parities[x]
        # Need py such that px ^ py = target
        # XOR of frozensets: symmetric difference
        py = px ^ target_set
        # Look for y in parity_map[py] where y != x and y > 210
        if py in parity_map:
            for y in parity_map[py]:
                if y == x:
                    continue
                if y <= 210:
                    continue
                max_xy = max(x, y)
                if max_xy < current_best_max:
                    # Verify
                    prod = x * y * 210
                    if _is_square(prod):
                        current_best_max = max_xy
                        best_raw = max_xy - 210
                        best_result = [x, y]
                        break  # For this x, we found the best y (smallest y first)
                else:
                    break  # y is getting larger, no point
    
    # Search for size 3: parity(x)^parity(y)^parity(z) = target
    # This is harder. Let's try small sets.
    # For size 3, we can fix x and y, then need z with parity = target ^ px ^ py.
    
    if best_raw > 100 and time.time() < deadline:
        # Try size 3 with small numbers
        small_numbers = list(range(211, min(500, max_val + 1)))
        for i in range(len(small_numbers)):
            if time.time() > deadline:
                break
            x = small_numbers[i]
            for j in range(i + 1, len(small_numbers)):
                if time.time() > deadline:
                    break
                y = small_numbers[j]
                needed_parity = target_set ^ parities[x] ^ parities[y]
                if needed_parity in parity_map:
                    for z in parity_map[needed_parity]:
                        if z == x or z == y or z <= 210:
                            continue
                        max_xyz = max(x, y, z)
                        if max_xyz < current_best_max:
                            prod = x * y * z * 210
                            if _is_square(prod):
                                current_best_max = max_xyz
                                best_raw = max_xyz - 210
                                best_result = [x, y, z]
                                break
                    else:
                        continue
                    break
    
    # Return as list of Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
