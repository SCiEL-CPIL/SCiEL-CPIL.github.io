import math
from itertools import product

def construct():
    # Approach: Local Search around the Primorial
    # We generate a set of candidate numbers based on the primorial of the first 15 primes
    # and its variations (swapping primes, adjusting exponents) and select the one
    # with the maximum tau_perp score.
    
    # Primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_score(n):
        # Calculate tau_perp(n)
        # Generate divisors
        divs = [1]
        for p in primes:
            if n % p == 0:
                # Determine exponent
                exp = 0
                temp = n
                while temp % p == 0:
                    temp //= p
                    exp += 1
                # Extend divisors
                new_divs = []
                for d in divs:
                    val = d
                    for _ in range(exp):
                        new_divs.append(val)
                        val *= p
                divs.extend(new_divs)
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Candidate 1: Primorial P_15
    n1 = 1
    for p in primes:
        n1 *= p
    s1 = get_score(n1)
    best_n = n1
    best_s = s1

    # Candidate 2: P_14 * 2^2 (Replace 47 with 2)
    n2 = 1
    for i, p in enumerate(primes):
        if p == 47:
            n2 *= 2 # Extra 2
        else:
            n2 *= p
    if n2 <= 10**18:
        s2 = get_score(n2)
        if s2 > best_s:
            best_s = s2
            best_n = n2

    # Candidate 3: P_14 * 3^2 (Replace 47 with 3)
    n3 = 1
    for i, p in enumerate(primes):
        if p == 47:
            n3 *= 3
        else:
            n3 *= p
    if n3 <= 10**18:
        s3 = get_score(n3)
        if s3 > best_s:
            best_s = s3
            best_n = n3

    # Candidate 4: P_15 / 47 * 53 (Replace 47 with 53)
    n4 = n1 // 47 * 53
    if n4 <= 10**18:
        s4 = get_score(n4)
        if s4 > best_s:
            best_s = s4
            best_n = n4

    # Candidate 5: P_13 * 2 * 3 * 5 * 7 (Replace 41, 43, 47 with 2, 3, 5, 7?)
    # This is getting complex. Let's stick to simple swaps.
    
    # Candidate 6: P_15 / 47 * 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    # This is P_14 * 2.
    n6 = n1 // 47 * 2
    if n6 <= 10**18:
        s6 = get_score(n6)
        if s6 > best_s:
            best_s = s6
            best_n = n6

    return best_n
