import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Generate primes up to 256 (max sum is 128+128=256)
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, 257) if is_prime[i]]
    prime_set = set(primes)

    # We look for a 3x3 grid of primes:
    # p1, p1+d1, p1+d2
    # p2, p2+d1, p2+d2
    # p3, p3+d1, p3+d2
    # where p1, p2, p3 are distinct primes.
    # This ensures all 9 sums are distinct primes, hence pairwise coprime.
    
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    best_score = 0

    def check_solution(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if len(sums) < best_score:
            return 0
            
        # Check pairwise coprime
        s_list = list(sums)
        for i in range(len(s_list)):
            for j in range(i+1, len(s_list)):
                if math.gcd(s_list[i], s_list[j]) != 1:
                    return 0
        return len(sums)

    # Initial check for hardcoded small solution if any
    # A=[1,2,3], B=[1,2,3] -> sums {2,3,4,5,6}. gcd(2,4)!=1. Invalid.
    
    # Search for prime grid
    # d1, d2 are differences in B.
    # We need 3 starting primes p such that p, p+d1, p+d2 are all prime.
    
    found = False
    for d1 in range(1, 128):
        if time.time() > deadline:
            break
        for d2 in range(1, 128):
            if d1 == d2:
                continue
            if time.time() > deadline:
                break
                
            # Find starting primes p such that p, p+d1, p+d2 are primes
            valid_starts = []
            for p in primes:
                if p + d1 > 256 or p + d2 > 256:
                    break
                if (p + d1) in prime_set and (p + d2) in prime_set:
                    valid_starts.append(p)
            
            if len(valid_starts) >= 3:
                # We have at least 3 starting primes. Pick the first 3.
                p1, p2, p3 = valid_starts[0], valid_starts[1], valid_starts[2]
                
                # Now find b1 such that A, B are in [1, 128]
                # B = {b1, b1+d1, b1+d2}
                # A = {p1-b1, p2-b1, p3-b1}
                
                # Constraints for b1:
                # 1 <= b1 <= 128
                # 1 <= b1+d1 <= 128  => 1-d1 <= b1 <= 128-d1
                # 1 <= b1+d2 <= 128  => 1-d2 <= b1 <= 128-d2
                # 1 <= p1-b1 <= 128  => p1-128 <= b1 <= p1-1
                # 1 <= p2-b1 <= 128  => p2-128 <= b1 <= p2-1
                # 1 <= p3-b1 <= 128  => p3-128 <= b1 <= p3-1
                
                lower = max(1, 1-d1, 1-d2, p1-128, p2-128, p3-128)
                upper = min(128, 128-d1, 128-d2, p1-1, p2-1, p3-1)
                
                if lower <= upper:
                    # Pick a valid b1
                    b1 = lower
                    B = [b1, b1+d1, b1+d2]
                    A = [p1-b1, p2-b1, p3-b1]
                    
                    # Verify distinctness and range
                    if len(set(A)) == 3 and len(set(B)) == 3:
                        if all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B):
                            score = check_solution(A, B)
                            if score > best_score:
                                best_score = score
                                best_A = A
                                best_B = B
                                if best_score == 9:
                                    return [best_A, best_B]
                    found = True

    # If no prime grid found, fallback to random search or simple exhaustive on small range
    # But prime grids are very likely to exist.
    
    if best_score < 9:
        # Fallback: Random search
        import random
        for _ in range(10000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            score = check_solution(A, B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
                if best_score == 9:
                    return [best_A, best_B]

    return [best_A, best_B]
