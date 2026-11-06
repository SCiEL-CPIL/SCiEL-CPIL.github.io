import math
import time

def _sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:n + 1:step] = [False] * len(range(start, n + 1, step))
    return is_prime

def construct():
    t_start = time.time()
    DEADLINE = 75

    MAX_SUM = 256
    is_prime = _sieve(MAX_SUM)
    
    # We want to find A, B such that |A+B| is max and distinct sums are pairwise coprime.
    # Max |A+B| is 9.
    # We search for 9 distinct pairwise coprime sums.
    
    best_score = 0
    best_A = [1, 1, 1] # Placeholder, will be updated
    best_B = [1, 1, 1]
    
    # Precompute a quick check for pairwise coprime
    def is_pairwise_coprime(nums):
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(nums[i], nums[j]) > 1:
                    return False
        return True

    # Search d1, d2
    # To get 9 distinct sums, we generally need d1, d2 to be "independent".
    # If gcd(d1, d2) == 1, always 9 distinct.
    # If gcd(d1, d2) > 1, we check.
    
    # Heuristic: Try small d1, d2 first? Or those with gcd 1 first?
    # gcd 1 cases are impossible for all-prime, but possible for pairwise coprime (non-prime).
    # Example: 9 distinct odd numbers, pairwise coprime.
    
    # Let's iterate d1, d2
    for d1 in range(1, 64):
        if time.time() - t_start > DEADLINE:
            break
        for d2 in range(1, 64):
            if time.time() - t_start > DEADLINE:
                break
            
            # Check if 9 sums are distinct
            # Sums are a + i*d1 + b + j*d2
            # Distinctness depends on d1, d2
            # If d1 == d2, sums are a+b + (i+j)d1.
            # i+j ranges 0..4. 5 distinct.
            # We want 9.
            
            # Quick check:
            # The set of offsets is {i*d1 + j*d2}
            # If len(set(offsets)) < 9, skip?
            # Actually, we can just check the distinct sums later.
            
            a_max = 128 - 2 * d1
            b_max = 128 - 2 * d2
            if a_max < 1 or b_max < 1:
                continue
                
            # We can restrict a, b to make sums odd?
            # If d1, d2 are even, sums parity = a+b.
            # If d1, d2 are odd, sums parity varies.
            
            # Iterate a, b
            # To speed up, we can sample or use a smarter bound.
            # But 128*128*63*63 is ~ 6M iterations.
            # Each check is cheap (9 gcds).
            # 6M * 9 ops = 54M ops. Should be < 1s in Python if optimized?
            # Python is slower. 6M iterations might take 5-10s.
            # We have 75s.
            
            for a in range(1, a_max + 1):
                if time.time() - t_start > DEADLINE:
                    break
                A = [a, a + d1, a + 2 * d1]
                for b in range(1, b_max + 1):
                    B = [b, b + d2, b + 2 * d2]
                    
                    sums = [ai + bj for ai in A for bj in B]
                    distinct_sums = list(set(sums))
                    
                    # We want max cardinality.
                    # If we find 9, we can return immediately?
                    # We need to check pairwise coprime.
                    if len(distinct_sums) == 9:
                        if is_pairwise_coprime(distinct_sums):
                            return [A, B]
                    # If we find > best_score, update
                    # But we want 9.
    
    # If not found, fallback to random or previous best
    # The previous best was found by random.
    # Let's add a random fallback with a tighter loop.
    
    import random
    rng = random.Random(42)
    best_score = 0
    best_A = None
    best_B = None
    
    # Try to find score 9
    for _ in range(100000):
        if time.time() - t_start > DEADLINE:
            break
        A = rng.sample(range(1, 129), 3)
        B = rng.sample(range(1, 129), 3)
        sums = [ai + bj for ai in A for bj in B]
        distinct = list(set(sums))
        if len(distinct) == 9:
            if is_pairwise_coprime(distinct):
                return [A, B]
        
        # Track best
        if len(distinct) > best_score:
            if is_pairwise_coprime(distinct):
                best_score = len(distinct)
                best_A = A
                best_B = B
    
    if best_A:
        return [best_A, best_B]
        
    return [[1, 2, 3], [1, 2, 3]]
