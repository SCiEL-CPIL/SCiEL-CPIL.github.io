import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    MAX_SUM = 48 * 512  # 24576
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5)+1):
            if x % i == 0:
                return False
        return True
    
    def build_dp(S):
        """Build dp array where dp[k] is bitmask of sums using exactly k elements from S."""
        n = len(S)
        dp = [0] * (n + 2)
        dp[0] = 1
        for x in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        return dp
    
    def build_dp_excluding(S, excluded):
        """Build dp array where dp[k] is bitmask of sums using exactly k elements from S \ {excluded}."""
        S_filtered = [x for x in S if x != excluded]
        n = len(S_filtered)
        dp = [0] * (n + 2)
        dp[0] = 1
        for x in S_filtered:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        return dp
    
    def is_valid_set(S):
        """Check if S is a valid set: no element is the mean of any subset of at least 2 OTHER elements."""
        n = len(S)
        S_set = set(S)
        
        for a in S:
            # Check if a is the mean of any subset T of S \ {a} with |T| >= 2
            # sum(T) = |T| * a
            dp_excl = build_dp_excluding(S, a)
            for k in range(2, n):  # |T| = k, T subset of S\{a}, so k <= n-1
                target = k * a
                if target > MAX_SUM:
                    break
                if (dp_excl[k] >> target) & 1:
                    return False
        return True
    
    def try_add(x, S, dp_full):
        """
        Try to add x to S. Returns True if valid.
        dp_full: dp array for S (all elements).
        We need to check:
        1. x is not the mean of any subset of S with size >= 2
        2. No a in S becomes the mean of a subset of S U {x} with size >= 2 that doesn't include a
        """
        n = len(S)
        
        # 1. Check x is not mean of subset of S, size >= 2
        for k in range(2, n + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if (dp_full[k] >> target) & 1:
                return False
        
        # 2. Check no a in S becomes mean of subset of S U {x} that doesn't include a
        # Subsets of S U {x} not including a: either subset of S not including a, or subset of S\{a} U {x} not including a
        # Subsets of S not including a: already valid (S was valid)
        # Subsets of (S\{a}) U {x} not including a: T U {x} where T subset of S\{a}
        #   Size = |T| + 1 >= 2, so |T| >= 1
        #   Mean condition: (sum(T) + x) / (|T|+1) = a
        #   sum(T) = (|T|+1)*a - x
        #   Need T subset of S\{a} with |T| = k, sum(T) = (k+1)*a - x, for k >= 1
        
        for a in S:
            if a == x:
                continue
            dp_excl = build_dp_excluding(S, a)
            for k in range(1, n):  # |T| = k, T subset of S\{a}, k <= n-1
                target = (k + 1) * a - x
                if target < 0 or target > MAX_SUM:
                    continue
                if (dp_excl[k] >> target) & 1:
                    return False
        
        return True
    
    def greedy_order(order, max_size=48):
        """Greedily build a valid set from the given order."""
        S = []
        dp = [0] * (max_size + 2)
        dp[0] = 1
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 3:
                break
            
            if try_add(x, S, dp):
                S.append(x)
                # Update dp
                for k in range(len(S) - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
        
        return S
    
    def local_search(S, max_size=48, max_iters=50):
        """Improve S by trying to add elements or swap."""
        S = S[:]
        S_set = set(S)
        dp = build_dp(S)
        nums = list(range(1, 513))
        candidates = [x for x in nums if x not in S_set]
        rng = random.Random(42)
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try to add
            if len(S) < max_size:
                for x in rng.sample(candidates, min(30, len(candidates))):
                    if try_add(x, S, dp):
                        S.append(x)
                        S_set.add(x)
                        dp = build_dp(S)
                        candidates = [y for y in nums if y not in S_set]
                        improved = True
                        break
            
            # Try swap
            if len(S) < max_size:
                continue
                
            for i in range(len(S)):
                if time.time() > deadline - 3:
                    break
                removed = S[i]
                S_temp = S[:i] + S[i+1:]
                dp_temp = build_dp(S_temp)
                
                for x in rng.sample(candidates, min(15, len(candidates))):
                    if x == removed:
                        continue
                    if try_add(x, S_temp, dp_temp):
                        S_temp.append(x)
                        S = S_temp
                        S_set = set(S)
                        dp = build_dp(S)
                        candidates = [y for y in nums if y not in S_set]
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return S
    
    # Generate diverse orderings
    nums = list(range(1, 513))
    
    orderings = []
    
    # Ascending
    orderings.append(nums[:])
    
    # Descending
    orderings.append(nums[::-1])
    
    # Primes first
    primes = [x for x in nums if is_prime(x)]
    non_primes = [x for x in nums if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # Powers of 2 first
    powers = [2**i for i in range(10) if 2**i <= 512]
    others = [x for x in nums if x not in set(powers)]
    orderings.append(powers + others)
    
    # Fibonacci
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= 512:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = list(dict.fromkeys(fibs))
    others_fib = [x for x in nums if x not in set(fibs)]
    orderings.append(fibs + others_fib)
    
    # Random shuffles
    rng = random.Random(42)
    for _ in range(20):
        o = nums[:]
        rng.shuffle(o)
        orderings.append(o)
    
    best_set = []
    best_len = 0
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        s = greedy_order(order)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
        
        if len(s) < 48:
            s_improved = local_search(s)
            if len(s_improved) > best_len:
                best_set = s_improved
                best_len = len(s_improved)
        
        if best_len == 48:
            break
    
    return best_set
