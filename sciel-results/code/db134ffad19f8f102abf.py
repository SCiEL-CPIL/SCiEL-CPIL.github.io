import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (10s buffer)
    
    # First 30 primes - allows larger primes in combinations
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    best_n = 2
    best_score = 0
    
    def calc_score(prime_list):
        """Calculate tau_perp score for a square-free number with given prime factors."""
        n = 1
        for p in prime_list:
            n *= p
        if n > 10**18:
            return -1
        
        # Generate all divisors
        divs = [1]
        for p in prime_list:
            divs = divs + [d * p for d in divs]
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    # Check all subsets of size 15 from first 20 primes (high priority, fast)
    for combo in itertools.combinations(primes[:20], 15):
        if time.time() > deadline - 40:
            break
        n = 1
        for p in combo:
            n *= p
        if n > 10**18:
            continue
        s = calc_score(combo)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Check all subsets of size 14 from first 23 primes
    for combo in itertools.combinations(primes[:23], 14):
        if time.time() > deadline - 25:
            break
        n = 1
        for p in combo:
            n *= p
        if n > 10**18:
            continue
        s = calc_score(combo)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Check all subsets of size 13 from first 26 primes
    for combo in itertools.combinations(primes[:26], 13):
        if time.time() > deadline - 15:
            break
        n = 1
        for p in combo:
            n *= p
        if n > 10**18:
            continue
        s = calc_score(combo)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n
