import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Primes up to 53
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    def get_divisors(n):
        divs = [1]
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                new_divs = []
                power = 1
                while n % p == 0:
                    n //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            # Optimization: if n becomes 1, break
            if n == 1:
                break
        if n > 1:
            for d in divs[:]:
                divs.append(d * n)
        return sorted(divs)

    def calc_score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    candidates = []
    
    # Candidate 1: Product of first 15 primes (Square-free)
    n_15 = 1
    for p in primes[:15]:
        n_15 *= p
    if n_15 <= 10**18:
        candidates.append(n_15)
        
    # Candidate 2: Product of first 14 primes, with 2 squared
    n_14_sq2 = 2 * 1
    for p in primes[1:14]: # 3..43
        n_14_sq2 *= p
    n_14_sq2 *= 2 # So 2^2 * 3 * ... * 43
    if n_14_sq2 <= 10**18:
        candidates.append(n_14_sq2)
        
    # Candidate 3: Product of first 14 primes, with 3 squared
    n_14_sq3 = 2 * 1
    for p in primes[1:14]:
        n_14_sq3 *= p
    n_14_sq3 *= 3 # So 2 * 3^2 * ... * 43
    if n_14_sq3 <= 10**18:
        candidates.append(n_14_sq3)
        
    # Candidate 4: Product of first 13 primes, with 2 and 3 squared
    n_13_sq23 = 4 * 9 * 1
    for p in primes[2:13]: # 5..41
        n_13_sq23 *= p
    if n_13_sq23 <= 10**18:
        candidates.append(n_13_sq23)
        
    # Candidate 5: Product of first 13 primes, with 2 cubed
    n_13_cub2 = 8 * 1
    for p in primes[1:13]:
        n_13_cub2 *= p
    if n_13_cub2 <= 10**18:
        candidates.append(n_13_cub2)

    # Candidate 6: Product of first 13 primes, with 3 cubed
    n_13_cub3 = 2 * 27 * 1
    for p in primes[2:13]:
        n_13_cub3 *= p
    if n_13_cub3 <= 10**18:
        candidates.append(n_13_cub3)

    # Candidate 7: Product of first 12 primes, with 2 and 3 to higher powers?
    # Let's just stick to simple variations for now to ensure speed.
    
    best_n = candidates[0]
    best_score = -1
    
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n
