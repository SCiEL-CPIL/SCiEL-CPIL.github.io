"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute smallest prime factor for fast factorization up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors_fast(n):
        """Return set of prime factors using SPF table."""
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    # Precompute prime factors for all values 2..20000
    pf_cache = [frozenset() for _ in range(MAX_SUM + 1)]
    for v in range(2, MAX_SUM + 1):
        pf_cache[v] = prime_factors_fast(v)

    def count_primes_fast(nums):
        """Count distinct primes dividing all pairwise sums using cache."""
        primes = frozenset()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                primes = primes | pf_cache[s]
        return len(primes)

    def count_primes_with_set(nums):
        """Count and return the set of distinct primes."""
        primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                primes.update(pf_cache[s])
        return len(primes), primes

    best_count = float('inf')
    best_nums = None

    def try_update(nums):
        nonlocal best_count, best_nums
        c = count_primes_fast(nums)
        if c < best_count:
            best_count = c
            best_nums = nums[:]
        return c

    # === STRATEGY 1: Graph Clique Search ===
    # For a target prime set P, build graph where edge (u,v) exists if
    # all prime factors of (u+v) are in P. Find clique of size 12.
    
    target_prime_sets = [
        frozenset([2]),
        frozenset([2, 3]),
        frozenset([3]),
        frozenset([2, 5]),
        frozenset([3, 5]),
        frozenset([2, 3, 5]),
        frozenset([2, 7]),
        frozenset([3, 7]),
        frozenset([5, 7]),
        frozenset([2, 3, 7]),
        frozenset([2, 5, 7]),
        frozenset([3, 5, 7]),
        frozenset([2, 3, 5, 7]),
        frozenset([2, 3, 11]),
        frozenset([2, 5, 11]),
        frozenset([2, 3, 5, 11]),
    ]

    for target_p in target_prime_sets:
        if time.time() > deadline - 5:
            break
        
        # Build adjacency: for each u in 1..10000, which v>u have u+v P-smooth
        # Instead of full adjacency, use greedy clique building
        # For each starting vertex, greedily add compatible vertices
        
        # Precompute: for each sum s in 2..20000, is it P-smooth?
        smooth_sum = [False] * (MAX_SUM + 1)
        for s in range(2, MAX_SUM + 1):
            if pf_cache[s] <= target_p:
                smooth_sum[s] = True
        
        # Count how many pairs are compatible (rough density check)
        # If too sparse, skip
        compatible_count = 0
        for s in range(2, MAX_SUM + 1):
            if smooth_sum[s]:
                # Number of pairs (u,v) with u+v=s, 1<=u<v<=10000
                lo = max(1, s - 9999)
                hi = min(9999, s - 1)
                if hi >= lo:
                    compatible_count += (hi - lo + 1) // 2  # u < v means u < s/2
        
        # If density is too low, skip this prime set
        total_pairs = 10000 * 9999 // 2
        if compatible_count < total_pairs * 0.001:
            continue
        
        # Greedy clique search: try multiple starting points
        found_clique = False
        for start_val in range(1, min(100, 10001)):
            if time.time() > deadline - 5:
                break
            
            clique = [start_val]
            # Greedily add vertices compatible with all existing clique members
            for candidate in range(start_val + 1, 10001):
                if len(clique) >= 12:
                    break
                ok = True
                for existing in clique:
                    s = candidate + existing
                    if s > MAX_SUM or not smooth_sum[s]:
                        ok = False
                        break
                if ok:
                    clique.append(candidate)
            
            if len(clique) == 12:
                c = count_primes_fast(clique)
                if c < best_count:
                    best_count = c
                    best_nums = clique[:]
                    found_clique = True
                break  # Found a valid set for this target_p, move on
        
        # If greedy from small start didn't work, try from larger starts
        if not found_clique and time.time() < deadline - 10:
            for start_val in range(500, 5000, 100):
                if time.time() > deadline - 5:
                    break
                clique = [start_val]
                for candidate in range(start_val + 1, min(start_val + 5000, 10001)):
                    if len(clique) >= 12:
                        break
                    ok = True
                    for existing in clique:
                        s = candidate + existing
                        if s > MAX_SUM or not smooth_sum[s]:
                            ok = False
                            break
                    if ok:
                        clique.append(candidate)
                
                if len(clique) == 12:
                    c = count_primes_fast(clique)
                    if c < best_count:
                        best_count = c
                        best_nums = clique[:]
                    break

    # === STRATEGY 2: Structured candidates (fallback) ===
    # All multiples of a smooth base B
    smooth_bases = set()
    for a in range(0, 14):
        v2 = 1 << a
        if v2 > 10000: break
        for b in range(0, 10):
            v = v2 * (3 ** b)
            if v > 10000: break
            for c in range(0, 7):
                v2_ = v * (5 ** c)
                if v2_ > 10000: break
                for d in range(0, 5):
                    v3 = v2_ * (7 ** d)
                    if v3 > 10000: break
                    smooth_bases.add(v3)
    
    smooth_bases = sorted(smooth_bases)

    for base in smooth_bases:
        if time.time() > deadline - 5:
            break
        max_q = 10000 // base
        if max_q < 12:
            continue
        for start_q in range(1, min(max_q - 11, 20) + 1):
            if time.time() > deadline - 5:
                break
            qs = list(range(start_q, start_q + 12))
            nums = [q * base for q in qs]
            try_update(nums)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            qs = random.sample(range(1, max_q + 1), 12)
            nums = [q * base for q in qs]
            try_update(nums)

    # === STRATEGY 3: Congruence class approach ===
    # If all nums ≡ r (mod M) and M | 2r, then all sums ≡ 0 (mod M)
    for M in [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 56, 60, 64, 70, 72, 80, 84, 90, 96, 100, 108, 112, 120, 128, 140, 144, 150, 160, 168, 180, 192, 200, 210, 216, 224, 240, 252, 256, 270, 280, 288, 300, 320, 336, 360, 384, 400, 420, 432, 448, 480, 500, 504, 512, 540, 560, 576, 600, 630, 640, 672, 700, 720, 750, 768, 800, 840, 864, 900, 960, 1000]:
        if time.time() > deadline - 5:
            break
        r = M // 2
        if r < 1:
            continue
        last = r + 11 * M
        if last > 10000:
            continue
        nums = [r + k*M for k in range(12)]
        try_update(nums)

    # === STRATEGY 4: Local improvement with incremental evaluation ===
    def count_primes_incremental(nums, i, old_val, new_val):
        """Compute new prime count after changing nums[i] from old_val to new_val."""
        primes = set()
        n = len(nums)
        for a in range(n):
            if a == i:
                continue
            for b in range(a+1, n):
                if b == i:
                    continue
                s = nums[a] + nums[b]
                primes.update(pf_cache[s])
        for j in range(n):
            if j == i:
                continue
            s = new_val + nums[j]
            primes.update(pf_cache[s])
        return len(primes)

    if best_nums is not None and time.time() < deadline - 10:
        nums = best_nums[:]
        c = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = nums[i]
                for delta in list(range(-30, 31)) + list(range(50, 151, 10)) + list(range(-150, -50, 10)):
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in nums:
                        continue
                    nums[i] = new_val
                    c2 = count_primes_incremental(nums, i, old_val, new_val)
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i] = old_val

    # === STRATEGY 5: Random restarts with local improvement ===
    if time.time() < deadline - 15:
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            mode = random.randint(0, 3)
            if mode == 0:
                base = random.choice([2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,32,35,36,40,42,45,48,50,54,56,60,63,64,70,72,75,80,84,90,96,100,105,108,112,120,126,128,135,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,315,320,336,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000])
                max_q = 10000 // base
                if max_q < 12:
                    continue
                qs = random.sample(range(1, max_q + 1), 12)
                nums = [q * base for q in qs]
            elif mode == 1:
                d = random.choice([2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,32,35,36,40,42,45,48,50,54,56,60,63,64,70,72,75,80,84,90,96,100,105,108,112,120,126,128,135,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,315,320,336,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000])
                max_a = 10000 - 11 * d
                if max_a < 1:
                    continue
                a = random.randint(1, max_a)
                nums = [a + i*d for i in range(12)]
            elif mode == 2:
                M = random.choice([4,6,8,10,12,14,16,18,20,24,28,30,32,36,40,42,48,56,60,64,70,72,80,84,90,96,100,108,112,120,128,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,320,336,360,384,400,420,432,448,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000])
                r = M // 2
                if r < 1:
                    continue
                last = r + 11 * M
                if last > 10000:
                    continue
                start_k = random.randint(0, max(0, (10000 - r) // M - 11))
                nums = [r + (start_k + k)*M for k in range(12)]
            else:
                nums = random.sample(range(1, 10001), 12)
            
            c = count_primes_fast(nums)
            if c <= best_count + 2:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(12):
                        if time.time() > deadline - 5:
                            break
                        old_val = nums[i]
                        for delta in range(-15, 16):
                            if delta == 0:
                                continue
                            new_val = old_val + delta
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val in nums:
                                continue
                            nums[i] = new_val
                            c2 = count_primes_incremental(nums, i, old_val, new_val)
                            if c2 < c:
                                c = c2
                                if c < best_count:
                                    best_count = c
                                    best_nums = nums[:]
                                improved = True
                            else:
                                nums[i] = old_val

    if best_nums is None:
        return list(range(1, 13))
    
    return best_nums
# EVOLVE-BLOCK-END
