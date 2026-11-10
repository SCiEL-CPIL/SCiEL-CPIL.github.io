import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def is_valid_candidate(x, S, dp):
        """
        Check if x can be added to S.
        S: list of currently selected integers.
        dp: list of integers, where dp[k] is a bitmask of possible sums using exactly k elements from S.
        """
        n = len(S)
        
        # 1. Check if x is the mean of any subset of S of size k >= 2.
        # sum(subset) = k * x
        for k in range(2, n + 1):
            target = k * x
            if target > 512 * k:  # Max possible sum
                break
            if (dp[k] >> target) & 1:
                return False
        
        # 2. Check if any a in S becomes the mean of a subset of S U {x} of size >= 2
        # that includes x.
        for a in S:
            for k in range(1, n + 1):
                target = (k + 1) * a - x
                if target < 0 or target > 512 * k:
                    continue
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def try_ordering(order):
        S = []
        dp = [0] * 49
        dp[0] = 1
        
        for x in order:
            if len(S) >= 48:
                break
            if is_valid_candidate(x, S, dp):
                S.append(x)
                # Update DP: iterate backwards to avoid using x multiple times
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
        
        return S

    # Generate diverse orderings
    nums = list(range(1, 513))
    
    # Ordering 1: Ascending
    ord_asc = nums[:]
    
    # Ordering 2: Descending
    ord_desc = nums[::-1]
    
    # Ordering 3: Primes first
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in nums if is_prime(x)]
    non_primes = [x for x in nums if not is_prime(x)]
    ord_prime = primes + non_primes
    
    # Ordering 4: Alternating small/large
    alt = []
    for i in range(256):
        if i + 1 <= 512:
            alt.append(i + 1)
        if 512 - i >= 1:
            alt.append(512 - i)
    ord_alt = alt
    
    # Ordering 5: Spaced out (pick every ~10th number, then fill gaps)
    spaced = list(range(1, 513, 10))
    remaining = [x for x in nums if x not in set(spaced)]
    ord_spaced = spaced + remaining
    
    # Ordering 6: Co-prime bias (numbers with few small factors first)
    def factor_score(x):
        score = 0
        for p in [2, 3, 5, 7, 11]:
            if x % p == 0:
                score += 1
        return score
    ord_coprime = sorted(nums, key=lambda x: (factor_score(x), x))
    
    # Ordering 7: Random with fixed seed
    rng = random.Random(12345)
    ord_rand1 = nums[:]
    rng.shuffle(ord_rand1)
    
    # Ordering 8: Another random
    rng2 = random.Random(99999)
    ord_rand2 = nums[:]
    rng2.shuffle(ord_rand2)
    
    # Ordering 9: Start with high numbers, then low
    ord_high_low = list(range(512, 0, -1)) + list(range(1, 513))
    
    # Ordering 10: Start with low numbers, then high
    ord_low_high = list(range(1, 513)) + list(range(512, 0, -1))
    
    ords = [ord_asc, ord_desc, ord_prime, ord_alt, ord_spaced, ord_coprime,
            ord_rand1, ord_rand2, ord_high_low, ord_low_high]
    
    # Add more random orderings if time permits
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline - 10:
            break
        rng_extra = random.Random(seed * 7919)
        o = nums[:]
        rng_extra.shuffle(o)
        ords.append(o)
    
    best_set = []
    
    for o in ords:
        if time.time() > deadline - 3:
            break
        s = try_ordering(o)
        if len(s) > len(best_set):
            best_set = s
        if len(best_set) == 48:
            break
    
    return best_set
