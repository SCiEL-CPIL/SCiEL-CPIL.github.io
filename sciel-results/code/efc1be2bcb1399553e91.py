"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return sorted(divs)
    
    def tau_perp_from_factors(factors):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors_from_factors(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    best_n = 0
    best_score = 0
    
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT or n < 2:
            return
        score = tau_perp_from_factors(factors)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Try squarefree with k primes for k = 13, 14, 15, 16
    for k in [13, 14, 15, 16]:
        try_candidate([(p, 1) for p in primes[:k]])
    
    # Try increasing exponents of small primes while keeping many primes
    # 15 primes: 2^a * 3 * 5 * 7 * ... * 47
    for a in range(2, 8):
        n = 2**a
        for p in primes[1:15]:
            n *= p
        if n <= LIMIT:
            try_candidate([(2, a)] + [(p, 1) for p in primes[1:15]])
    
    # 14 primes: 2^a * 3^b * 5 * 7 * ... * 43
    for a in range(2, 10):
        for b in range(1, 5):
            n = 2**a * 3**b
            for p in primes[2:14]:
                n *= p
            if n <= LIMIT:
                try_candidate([(2, a), (3, b)] + [(p, 1) for p in primes[2:14]])
    
    # 13 primes: 2^a * 3^b * 5^c * 7 * ... * 41
    for a in range(2, 12):
        for b in range(1, 6):
            for c in range(1, 4):
                n = 2**a * 3**b * 5**c
                for p in primes[3:13]:
                    n *= p
                if n <= LIMIT:
                    try_candidate([(2, a), (3, b), (5, c)] + [(p, 1) for p in primes[3:13]])
    
    # 12 primes: 2^a * 3^b * 5^c * 7^d * 11 * ... * 37
    for a in range(2, 14):
        for b in range(1, 7):
            for c in range(1, 5):
                for d in range(1, 4):
                    n = 2**a * 3**b * 5**c * 7**d
                    for p in primes[4:12]:
                        n *= p
                    if n <= LIMIT:
                        try_candidate([(2, a), (3, b), (5, c), (7, d)] + [(p, 1) for p in primes[4:12]])
    
    # 11 primes: 2^a * 3^b * 5^c * 7^d * 11^e * 13 * ... * 31
    for a in range(2, 16):
        for b in range(1, 8):
            for c in range(1, 6):
                for d in range(1, 5):
                    for e in range(1, 4):
                        n = 2**a * 3**b * 5**c * 7**d * 11**e
                        for p in primes[5:11]:
                            n *= p
                        if n <= LIMIT:
                            try_candidate([(2, a), (3, b), (5, c), (7, d), (11, e)] + [(p, 1) for p in primes[5:11]])
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # 10 primes: 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17 * ... * 29
    for a in range(2, 18):
        for b in range(1, 9):
            for c in range(1, 7):
                for d in range(1, 6):
                    for e in range(1, 5):
                        for f in range(1, 4):
                            if time.time() - start_time > DEADLINE:
                                return best_n
                            n = 2**a * 3**b * 5**c * 7**d * 11**e * 13**f
                            for p in primes[6:10]:
                                n *= p
                            if n <= LIMIT:
                                try_candidate([(2, a), (3, b), (5, c), (7, d), (11, e), (13, f)] + [(p, 1) for p in primes[6:10]])
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # 9 primes: 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19 * 23
    for a in range(2, 20):
        for b in range(1, 10):
            for c in range(1, 8):
                for d in range(1, 7):
                    for e in range(1, 6):
                        for f in range(1, 5):
                            for g in range(1, 4):
                                if time.time() - start_time > DEADLINE:
                                    return best_n
                                n = 2**a * 3**b * 5**c * 7**d * 11**e * 13**f * 17**g
                                for p in primes[7:9]:
                                    n *= p
                                if n <= LIMIT:
                                    try_candidate([(2, a), (3, b), (5, c), (7, d), (11, e), (13, f), (17, g)] + [(p, 1) for p in primes[7:9]])
    
    return best_n
# EVOLVE-BLOCK-END
