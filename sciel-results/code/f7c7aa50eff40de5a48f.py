"""Search for n in [1000, 1000000] minimizing distinct small-prime factors across 24 consecutive numbers."""
import time
import math

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
T = 24
MIN_N = 1000
MAX_N = 1000000
DEADLINE = 85.0  # safety margin before 90s kill


def small_prime_factor_count(m):
    """Count distinct prime factors of m that are strictly below 24."""
    cnt = 0
    for p in PRIMES:
        if m % p == 0:
            cnt += 1
            while m % p == 0:
                m //= p
    return cnt


def score_for_n(n):
    """For each n+i (1<=i<=24), compute the number of distinct small-prime factors.
    The total score is the sum of these counts (or we could use max, but let's read carefully).
    
    Re-reading: 'extract its complete factor composed of primes STRICTLY BELOW 24. MINIMIZE the number of distinct extracted factors.'
    
    This means for each n+i, we extract the part that is composed of primes < 24. 
    The 'distinct extracted factors' likely means the set of distinct primes < 24 that appear in the factorizations of all 24 numbers.
    
    Wait, let me re-read: 'For each n+i, 1<=i<=24, extract its complete factor composed of primes STRICTLY BELOW 24. MINIMIZE the number of distinct extracted factors.'
    
    I think it means: across all 24 numbers n+1 through n+24, look at all the small prime factors (primes < 24) that appear in their factorizations, and count how many distinct such primes there are. Minimize that count.
    
    Actually, another interpretation: for each n+i, the "complete factor composed of primes < 24" is the product of all prime powers p^k where p<24 and p divides n+i. The "distinct extracted factors" would be the distinct prime bases across all 24 numbers.
    
    So the score is: |{p in PRIMES : p divides n+i for some i in 1..24}|
    
    Let me reconsider. The raw value is what's measured, lower is better. The seed returns 1000 with score 1.0. Let me think about what score 1000 gives.
    
    For n=1000, the numbers are 1001..1024. Let me check which small primes appear:
    - 2: 1002, 1004, etc. - yes
    - 3: 1002, 1005, etc. - yes  
    - 5: 1005, 1010, etc. - yes
    - 7: 1001=7*11*13, 1015=7*5*29 - yes
    - 11: 1001=7*11*13 - yes
    - 13: 1001=7*11*13 - yes
    - 17: 1020=17*60 - yes
    - 19: ? 1007=19*53 - yes
    - 23: 1012=23*44 - yes
    
    So for n=1000, all 9 small primes appear. Score would be 9.
    
    The best possible score is... well, we need at least some primes to appear. The minimum could be as low as 1 or 2 if we find a stretch where only 1-2 small primes divide any of the 24 numbers.
    
    Actually wait - every even number is divisible by 2, and in any 24 consecutive numbers, there are 12 even numbers. So 2 will ALWAYS appear. Similarly, in any 24 consecutive numbers, there are 8 multiples of 3, so 3 always appears. 5: in any 24 consecutive, there are at least 4 multiples of 5. So 2, 3, 5 always appear. 7: in any 24 consecutive, there are at least 3 multiples of 7. So 7 always appears too.
    
    So the minimum is at least 4 (primes 2,3,5,7 always appear). Can we get 4? We'd need a stretch of 24 numbers where none of 11,13,17,19,23 divide any of the numbers.
    
    Let me search for such n."""
    # For each prime p, check if any n+i is divisible by p
    distinct_primes = set()
    for p in PRIMES:
        # n+i divisible by p for some i in 1..24 means (n+i) % p == 0
        # i.e., n % p == -i % p == (p - i%p) % p for some i in 1..24
        # The residues n%p that work: {(p - i % p) % p for i in 1..24}
        # Since 24 >= p for all our primes (largest is 23, 24>=23), 
        # for p<=23, the set of i%p for i=1..24 covers all residues 0..p-1 (since 24>=p)
        # Actually for p=23: i=1..24 gives residues 1,2,...,22,0,1 - covers all
        # For p<=24, 24 consecutive integers cover all residue classes mod p
        # So EVERY n has at least one n+i divisible by p for any p <= 24
        # This means ALL 9 primes ALWAYS appear!
        
        # Wait, that can't be right. Let me verify with p=2: 24 consecutive numbers include 12 even, yes.
        # p=3: 8 multiples, yes. p=5: 4 or 5 multiples, yes. p=7: 3 or 4, yes.
        # p=11: 2 or 3, yes. p=13: 1 or 2, yes. p=17: 1, yes. p=19: 1, yes. p=23: 1, yes.
        
        # So for ANY n, all 9 primes < 24 divide at least one of n+1..n+24.
        # The score is always 9?!
        
        # Hmm, but the task says "minimize" implying it varies. Let me re-read.
        
        # "extract its complete factor composed of primes STRICTLY BELOW 24"
        # Maybe it's per-number: for each n+i, extract the small-prime part, and the score is 
        # the number of distinct primes used across ALL the extracted factors.
        # But as I showed, it's always all 9.
        
        # OR maybe the score is the SUM over i of (number of distinct small prime factors of n+i)?
        # That would vary with n.
        
        # Let me go with: score = sum over i=1..24 of (count of distinct primes < 24 dividing n+i)
        pass
    
    total = 0
    for i in range(1, T + 1):
        m = n + i
        for p in PRIMES:
            if m % p == 0:
                total += 1
                while m % p == 0:
                    m //= p
    return total


def construct():
    start_time = time.time()
    
    # First, quick check: is the score always 9 (count of distinct primes) or does it vary?
    # Let me just compute with the sum interpretation and search.
    
    best_n = MIN_N
    best_score = score_for_n(MIN_N)
    
    # Search in a structured way
    # Try to find n where the numbers n+1..n+24 have few small prime factors on average
    
    deadline = start_time + DEADLINE
    step = 1
    
    # First pass: scan with step 1 in a moderate range to get a feel
    # Then refine
    n = MIN_N
    while n <= MAX_N:
        if time.time() > deadline - 5:
            break
        s = score_for_n(n)
        if s < best_score:
            best_score = s
            best_n = n
            # Found a better one, let's search around it more carefully
            # But for now, keep scanning
        n += 1
    
    # If we have time, do a local refinement around the best
    remaining = deadline - time.time()
    if remaining > 2 and best_n > MIN_N + 1000:
        lo = max(MIN_N, best_n - 2000)
        hi = min(MAX_N, best_n + 2000)
        for n in range(lo, hi + 1):
            if time.time() > deadline:
                break
            s = score_for_n(n)
            if s < best_score:
                best_score = s
                best_n = n
    
    return best_n
