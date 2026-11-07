import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def sf_kernel(n):
        """Return squarefree kernel of n."""
        if n <= 1:
            return 1
        result = 1
        f = n
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2
        if f > 1:
            result *= f
        return result
    
    def is_perfect_square(n):
        if n < 0:
            return False
        x = math.isqrt(n)
        return x * x == n
    
    # Precompute squarefree kernels up to 10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)
    
    # Build map from sf value to sorted list of numbers with that sf
    sf_to_nums = defaultdict(list)
    for n in range(211, max_val + 1):
        sf_to_nums[sf[n]].append(n)
    
    # Also build sf_to_min for quick lookup
    sf_to_min = {}
    for s, nums in sf_to_nums.items():
        sf_to_min[s] = nums[0]
    
    # Precompute prime factor sets for each squarefree kernel value
    # Map: frozenset of primes -> squarefree kernel value
    # And: squarefree kernel value -> frozenset of primes
    sf_to_primes = {}
    primes_to_sf = {}
    for s in sf_to_nums.keys():
        n = s
        primes = frozenset()
        d = 2
        tmp = s
        while d * d <= tmp:
            if tmp % d == 0:
                primes = primes | {d}
                while tmp % d == 0:
                    tmp //= d
            d += 1 if d == 2 else 2
        if tmp > 1:
            primes = primes | {tmp}
        sf_to_primes[s] = primes
        primes_to_sf[primes] = s
    
    # Target: squarefree kernel of 210 = 210
    target = 210
    target_primes = sf_to_primes[target]
    
    best_max = max_val + 1
    best_list = None
    
    # Size 1: find n in [211, 10210] with sf[n] == 210
    if target in sf_to_min:
        n = sf_to_min[target]
        best_max = n
        best_list = [n]
    
    # Size 2: need sf[a] and sf[b] such that sf[a] XOR sf[b] (in prime set sense) = target
    # i.e., sf_to_primes[sf[a]] ^ sf_to_primes[sf[b]] = target_primes
    # So sf_to_primes[sf[b]] = target_primes ^ sf_to_primes[sf[a]]
    
    # Build list of (a, sf[a], primes_of_sf[a]) for a in [211, max_val]
    # For size 2, iterate over a, compute required primes for b, look up
    
    for a in range(211, max_val + 1):
        if a >= best_max:
            break
        sa = sf[a]
        pa = sf_to_primes[sa]
        required_pb = target_primes ^ pa
        if required_pb in primes_to_sf:
            sb = primes_to_sf[required_pb]
            if sb in sf_to_min:
                b = sf_to_min[sb]
                if b != a and b < best_max:
                    m = max(a, b)
                    if m < best_max:
                        best_max = m
                        best_list = [a, b]
    
    # Size 3: For pair (a,b), need c such that target_primes ^ pa ^ pb ^ pc = empty set
    # So pc = target_primes ^ pa ^ pb
    # Look up if pc corresponds to a valid sf value with a number c
    
    search_limit = min(best_max, 5000)
    
    # Precompute for each a in range: (a, sf[a], primes)
    # For faster access, just use arrays
    
    for a in range(211, min(search_limit, 3000)):
        if time.time() > deadline:
            break
        sa = sf[a]
        pa = sf_to_primes[sa]
        for b in range(a + 1, min(search_limit, 3000)):
            if time.time() > deadline:
                break
            sb = sf[b]
            pb = sf_to_primes[sb]
            required_pc = target_primes ^ pa ^ pb
            if required_pc in primes_to_sf:
                sc = primes_to_sf[required_pc]
                if sc in sf_to_min:
                    c = sf_to_min[sc]
                    if c != a and c != b and c < best_max:
                        m = max(a, b, c)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b, c]
    
    # Try larger range if time permits
    if time.time() < deadline - 20:
        for a in range(211, min(search_limit, 5000)):
            if time.time() > deadline - 10:
                break
            sa = sf[a]
            pa = sf_to_primes[sa]
            for b in range(a + 1, min(search_limit, 5000)):
                if time.time() > deadline - 10:
                    break
                sb = sf[b]
                pb = sf_to_primes[sb]
                required_pc = target_primes ^ pa ^ pb
                if required_pc in primes_to_sf:
                    sc = primes_to_sf[required_pc]
                    if sc in sf_to_min:
                        c = sf_to_min[sc]
                        if c != a and c != b and c < best_max:
                            m = max(a, b, c)
                            if m < best_max:
                                best_max = m
                                best_list = [a, b, c]
    
    # Size 4: For triple (a,b,c), need d such that target_primes ^ pa ^ pb ^ pc ^ pd = empty
    # pd = target_primes ^ pa ^ pb ^ pc
    # This is O(M^3) which is too slow for large M. Try small M.
    
    if time.time() < deadline - 30 and best_max > 220:
        # Try size 4 with small range
        small_limit = min(best_max, 500)
        for a in range(211, small_limit):
            if time.time() > deadline - 15:
                break
            sa = sf[a]
            pa = sf_to_primes[sa]
            for b in range(a + 1, small_limit):
                if time.time() > deadline - 15:
                    break
                sb = sf[b]
                pb = sf_to_primes[sb]
                for c in range(b + 1, small_limit):
                    if time.time() > deadline - 15:
                        break
                    sc = sf[c]
                    pc = sf_to_primes[sc]
                    required_pd = target_primes ^ pa ^ pb ^ pc
                    if required_pd in primes_to_sf:
                        sd = primes_to_sf[required_pd]
                        if sd in sf_to_min:
                            d = sf_to_min[sd]
                            if d != a and d != b and d != c and d < best_max:
                                m = max(a, b, c, d)
                                if m < best_max:
                                    best_max = m
                                    best_list = [a, b, c, d]
    
    if best_list is None:
        best_list = [840]
        best_max = 840
    
    # Verify the result
    prod = 210
    for x in best_list:
        prod *= x
    if not is_perfect_square(prod):
        # Fallback to known valid solution
        best_list = [840]
        best_max = 840
    
    # Ensure all values are Python ints
    result = [int(x) for x in best_list]
    return result
