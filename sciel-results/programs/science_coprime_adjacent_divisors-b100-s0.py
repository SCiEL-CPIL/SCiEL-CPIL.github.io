import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_divisors(n):
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if temp == 1:
                break
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
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

    # Candidate: 2^5 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    n_best = (2**5) * (3**2)
    for p in primes[2:14]:
        n_best *= p
        
    if n_best > 10**18:
        # Fallback to square-free 15 primes
        n_best = 1
        for p in primes[:15]:
            n_best *= p

    # Verify and potentially search neighbors
    best_n = n_best
    best_score = calc_score(n_best)
    
    # Try a few variations to ensure optimality
    candidates = [n_best]
    
    # 2^4 * 3^3 * 5 * ... * 43
    n2 = (2**4) * (3**3)
    for p in primes[2:14]:
        n2 *= p
    if n2 <= 10**18:
        candidates.append(n2)
        
    # 2^6 * 3^1 * 5 * ... * 43
    n3 = (2**6) * (3**1)
    for p in primes[2:14]:
        n3 *= p
    if n3 <= 10**18:
        candidates.append(n3)
        
    # 2^5 * 3^1 * 5^2 * 7 * ... * 43
    n4 = (2**5) * (3**1) * (5**2)
    for p in primes[3:14]:
        n4 *= p
    if n4 <= 10**18:
        candidates.append(n4)
        
    # 2^3 * 3^4 * 5 * ... * 43
    n5 = (2**3) * (3**4)
    for p in primes[2:14]:
        n5 *= p
    if n5 <= 10**18:
        candidates.append(n5)
        
    # 2^2 * 3^5 * 5 * ... * 43
    n6 = (2**2) * (3**5)
    for p in primes[2:14]:
        n6 *= p
    if n6 <= 10**18:
        candidates.append(n6)
        
    # 2^1 * 3^6 * 5 * ... * 43
    n7 = (2**1) * (3**6)
    for p in primes[2:14]:
        n7 *= p
    if n7 <= 10**18:
        candidates.append(n7)

    # 2^5 * 3^2 * 5 * 7 * ... * 47 (15 primes) - likely too big
    n8 = (2**5) * (3**2)
    for p in primes[2:15]:
        n8 *= p
    if n8 <= 10**18:
        candidates.append(n8)
        
    # 2^4 * 3^2 * 5 * 7 * ... * 47
    n9 = (2**4) * (3**2)
    for p in primes[2:15]:
        n9 *= p
    if n9 <= 10**18:
        candidates.append(n9)

    # 2^3 * 3^2 * 5 * 7 * ... * 47
    n10 = (2**3) * (3**2)
    for p in primes[2:15]:
        n10 *= p
    if n10 <= 10**18:
        candidates.append(n10)

    # 2^2 * 3^2 * 5 * 7 * ... * 47
    n11 = (2**2) * (3**2)
    for p in primes[2:15]:
        n11 *= p
    if n11 <= 10**18:
        candidates.append(n11)
        
    # 2^1 * 3^2 * 5 * 7 * ... * 47
    n12 = (2**1) * (3**2)
    for p in primes[2:15]:
        n12 *= p
    if n12 <= 10**18:
        candidates.append(n12)
        
    # 2^1 * 3^1 * 5 * 7 * ... * 47 (Square free 15)
    n13 = 1
    for p in primes[:15]:
        n13 *= p
    if n13 <= 10**18:
        candidates.append(n13)

    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n

    return best_n
