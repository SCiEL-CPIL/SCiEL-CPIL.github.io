import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    # 110s deadline to be safe
    deadline = start_time + 110.0

    def generate_powerful(limit):
        """
        Generates powerful numbers up to 'limit'.
        Powerful numbers are of the form a^2 * b^3 where b is square-free.
        """
        powerful = set()
        # b^3 <= limit => b <= limit^(1/3)
        max_b = int(limit ** (1.0/3.0)) + 2
        
        # We need to check if b is square-free.
        # For small limits, simple trial division is okay.
        # But we can optimize by precomputing primes up to sqrt(limit^(1/3)) = limit^(1/6).
        # limit max 10^18 => limit^(1/6) = 10^3 = 1000. Very fast.
        
        # Actually, we can just generate square-free numbers up to max_b.
        # A number n is square-free if it is not divisible by p^2 for any prime p.
        
        # Let's generate primes up to sqrt(max_b) for square-free check?
        # No, to check if b is square-free, we need to check divisibility by p^2 for p up to sqrt(b).
        # Since b <= 10^6 (if limit=10^18), sqrt(b) <= 1000.
        
        # Precompute squares of primes up to 1000
        def get_prime_squares(limit):
            is_prime = [True] * (limit + 1)
            is_prime[0] = False
            if limit >= 1:
                is_prime[1] = False
            for i in range(2, int(limit ** 0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            squares = [i*i for i in range(2, limit + 1) if is_prime[i]]
            return squares
        
        max_b_sqrt = int(max_b ** 0.5) + 2
        prime_squares = get_prime_squares(max_b_sqrt)
        
        def is_square_free(n):
            if n < 2:
                return True
            for ps in prime_squares:
                if ps > n:
                    break
                if n % ps == 0:
                    return False
            return True

        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            # a^2 * b^3 <= limit => a <= sqrt(limit / b^3)
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        
        return sorted(powerful)

    # Strategy:
    # 1. Generate a large set of powerful numbers P.
    #    To find many representations for N, N should be a sum of two "common" powerful numbers.
    #    The density of powerful numbers is low, but they are clustered at small values.
    #    However, sums of two small powerful numbers are small.
    #    Sums of two large powerful numbers are large but sparse.
    #    
    #    The "hot" N values are likely formed by sums of powerful numbers that are not too small and not too large.
    #    
    #    Let's generate powerful numbers up to L = 10^12.
    #    M(L) ~ L^(1/3) * something. 
    #    Number of powerful numbers up to X is ~ X^(1/3) * C.
    #    For X = 10^12, M ~ 10^4 * C. Let's say M ~ 50,000.
    #    O(M^2) is 2.5 * 10^9. Too slow in Python.
    #    
    #    So we cannot do full O(M^2) counting for M=50,000.
    #    
    #    Alternative:
    #    Generate P up to L1 = 10^9. M ~ 1000. M^2 = 10^6. Fast.
    #    This finds N up to 2*10^9.
    #    But maybe a larger N has more reps?
    #    The conjecture is about subpolynomial bounds. 
    #    Usually, the maximum number of representations for N=a+b (powerful) grows slowly.
    #    Small Ns might have more reps because the powerful numbers are denser relative to N?
    #    Actually, the count of representations r(N) is roughly constant on average? No, it varies.
    #    
    #    Let's try to generate P up to L = 10^10.
    #    M ~ 10^(10/3) ~ 2*10^3? No.
    #    Count of powerful numbers <= X is ~ 0.8 * X^(1/3).
    #    For X = 10^10, M ~ 8000. M^2 = 6.4 * 10^7. This is feasible in Python if optimized (NumPy or C-like loops? No, pure Python is ~10^7 ops/s). 6.4 * 10^7 might take 10-20s.
    #    
    #    Let's use L = 10^10.
    #    Generate P.
    #    Use a dictionary to count sums? 
    #    If we use a set P, and iterate i, j in P:
    #    sum = P[i] + P[j]. 
    #    We can just count them.
    
    L = 10**10
    if time.time() > deadline - 60:
        # Fallback to smaller if generation is slow? 
        # Generation of 10^10 powerfuls:
        # b up to 2154. 
        # For each b, a up to sqrt(10^10 / b^3).
        # Total iterations ~ sum_{b square-free} sqrt(L/b^3).
        # ~ L^(1/2) * sum b^(-1.5) ~ 10^5 * 1.5 ~ 1.5 * 10^5.
        # This is VERY fast. < 0.1s.
        pass
        
    P = generate_powerful(L)
    P_set = set(P)
    
    # Now count sums.
    # M = len(P).
    # If M is ~ 8000, M^2/2 ~ 32 * 10^6.
    # Python loop: 32M iterations. ~ 5-10 seconds.
    
    counts = defaultdict(int)
    pairs_map = defaultdict(list)
    
    # Optimization: Only store pairs for the top candidates?
    # We don't know the top candidates until we count.
    # Storing all pairs is memory heavy if many sums are frequent.
    # But we can just count first, find top N, then re-extract pairs for that N.
    
    # Counting phase
    n_P = len(P)
    for i in range(n_P):
        a = P[i]
        # b must be >= a to keep a <= b
        # And a + b <= 2*L (max sum)
        for j in range(i, n_P):
            b = P[j]
            s = a + b
            counts[s] += 1
            if time.time() > deadline - 10:
                # Break out early if running out of time
                # We have partial counts. 
                # Find best N from partial counts.
                # This is risky but better than timeout.
                break
        if time.time() > deadline - 10:
            break
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find top N
    # Sort by count
    # counts is a dict. 
    # sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # This might be slow if counts is huge (up to 2*10^10 keys? No, sums are dense in [2, 2L]).
    # Number of unique sums is at most M^2/2.
    # Sorting 30M items is slow.
    
    # Better: Keep a heap or just find max?
    # We want the N with the MAX count.
    # We can just iterate counts and find max.
    
    best_N = 0
    best_count = 0
    for N, c in counts.items():
        if c > best_count:
            best_count = c
            best_N = N
            
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Now extract pairs for best_N
    pairs = []
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    # Verify distinctness and a<=b
    # P is sorted, so a is increasing. b = N-a is decreasing.
    # If a <= b, then a <= N/2.
    # We ensured a <= N/2.
    # Are pairs distinct?
    # Since a is unique in P, and b is determined by a, pairs are distinct.
    
    return {'N': best_N, 'pairs': pairs}
