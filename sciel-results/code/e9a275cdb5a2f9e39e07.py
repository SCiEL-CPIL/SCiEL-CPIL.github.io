import math
import time

# EVOLVE-BLOCK-START
def _get_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs

def _tau_perp(n):
    divs = _get_divisors(n)
    if len(divs) < 2:
        return 0
    score = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            score += 1
    return score

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    best_n = 2
    best_score = 0
    
    # Candidate family 1: Product of first k primes (squarefree)
    prod = 1
    for k in range(1, len(primes)):
        prod *= primes[k-1]
        if prod > 10**18:
            break
        score = _tau_perp(prod)
        if score > best_score:
            best_score = score
            best_n = prod
            
    # Candidate family 2: 2^a * product of next k odd primes
    # n = 2^a * (3 * 5 * ... * p_{k+1})
    for a in range(1, 60): # 2^a <= 10^18 -> a <= 59
        power2 = 1 << a
        if power2 > 10**18:
            break
        prod = 1
        for k in range(1, len(primes)-1):
            prod *= primes[k] # primes[1] is 3, primes[2] is 5, etc.
            n = power2 * prod
            if n > 10**18:
                break
            score = _tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
                
    return best_n
# EVOLVE-BLOCK-END
