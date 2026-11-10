import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    MAX_SUM = 48 * 512
    
    def is_valid_candidate(x, S, dp):
        n = len(S)
        for k in range(2, n + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if (dp[k] >> target) & 1:
                return False
        for a in S:
            for k in range(1, n):
                target = (k + 1) * a - x
                if target <= 0:
                    continue
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def build_dp(S):
        dp = [0] * 49
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
        return dp

    def try_ordering(order):
        S = []
        dp = [0] * 49
        dp[0] = 1
        for x in order:
            if len(S) >= 48:
                break
            if is_valid_candidate(x, S, dp):
                S.append(x)
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
        return S

    def local_search(S, time_limit):
        S = S[:]
        if not S:
            return S
        dp = build_dp(S)
        nums = list(range(1, 513))
        S_set = set(S)
        candidates = [x for x in nums if x not in S_set]
        
        for _ in range(30):
            if time.time() > time_limit:
                break
            if len(S) >= 48:
                break
            
            added = False
            for i, x in enumerate(candidates):
                if is_valid_candidate(x, S, dp):
                    S.append(x)
                    S_set.add(x)
                    candidates.pop(i)
                    for k in range(len(S) - 1, -1, -1):
                        dp[k + 1] |= (dp[k] << x)
                    added = True
                    break
            
            if added:
                continue
            
            swapped = False
            for i, a in enumerate(S):
                S.remove(a)
                S_set.discard(a)
                n2 = len(S)
                dp2 = build_dp(S)
                
                for j, x in enumerate(candidates):
                    if is_valid_candidate(x, S, dp2):
                        S.append(x)
                        S_set.add(x)
                        candidates.pop(j)
                        for k in range(len(S) - 1, -1, -1):
                            dp2[k + 1] |= (dp2[k] << x)
                        dp = dp2
                        swapped = True
                        break
                
                if not swapped:
                    S.append(a)
                    S_set.add(a)
                    dp = build_dp(S)
                else:
                    break
            
            if not swapped:
                break
        
        return S

    nums = list(range(1, 513))
    
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
    
    def factor_score(x):
        score = 0
        for p in [2, 3, 5, 7, 11, 13, 17]:
            if x % p == 0:
                score += 1
        return score
    
    ords = []
    ords.append(nums[:])
    ords.append(nums[::-1])
    ords.append(primes + non_primes)
    ords.append(sorted(nums, key=lambda x: (factor_score(x), x)))
    ords.append(sorted(nums, key=lambda x: (factor_score(x), -x)))
    
    spaced = list(range(1, 513, 11))
    remaining = [x for x in nums if x not in set(spaced)]
    ords.append(spaced + remaining)
    
    odd_nums = [x for x in nums if x % 2 == 1]
    even_nums = [x for x in nums if x % 2 == 0]
    ords.append(odd_nums + even_nums)
    ords.append(even_nums + odd_nums)
    ords.append(list(range(512, 0, -1)))
    
    for seed in [42, 12345, 99999, 7, 13, 21, 33, 55, 77, 99]:
        rng = random.Random(seed)
        o = nums[:]
        rng.shuffle(o)
        ords.append(o)
    
    best_set = []
    
    for o in ords:
        if time.time() > deadline - 15:
            break
        s = try_ordering(o)
        if len(s) > len(best_set):
            best_set = s
        if len(best_set) == 48:
            break
    
    if len(best_set) > 0 and time.time() < deadline - 5:
        best_set = local_search(best_set, deadline - 3)
    
    if len(best_set) < 48 and time.time() < deadline - 10:
        for seed in range(200, 300):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed)
            o = nums[:]
            rng.shuffle(o)
            s = try_ordering(o)
            if len(s) > len(best_set):
                best_set = s
                if len(best_set) == 48:
                    break
            if time.time() < deadline - 5:
                s2 = local_search(s, deadline - 3)
                if len(s2) > len(best_set):
                    best_set = s2
                    if len(best_set) == 48:
                        break
    
    return best_set
