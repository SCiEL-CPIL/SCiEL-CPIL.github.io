import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Primes up to ~100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def tau_perp(n):
        """Calculate the number of adjacent coprime pairs in divisors of n."""
        if n < 2:
            return 0
        divisors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
            i += 1
        divisors.sort()
        
        count = 0
        for j in range(len(divisors) - 1):
            if math.gcd(divisors[j], divisors[j+1]) == 1:
                count += 1
        return count

    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = tau_perp(best_n)
    
    # Candidate 1: Square-free 15 primes
    n_sqf = 1
    for p in primes[:15]:
        n_sqf *= p
    if n_sqf <= 10**18:
        score = tau_perp(n_sqf)
        if score > best_score:
            best_score = score
            best_n = n_sqf

    # Candidate 2: 2^3 * product of next 14 primes (5 to 47)
    n_cand2 = (2**3)
    for p in primes[2:16]: # 5 to 47 (indices 2 to 15 inclusive)
        n_cand2 *= p
    if n_cand2 <= 10**18:
        score = tau_perp(n_cand2)
        if score > best_score:
            best_score = score
            best_n = n_cand2

    # Candidate 3: 2^4 * 3 * product of next 12 primes (5 to 43)
    n_cand3 = (2**4) * 3
    for p in primes[2:14]: # 5 to 43
        n_cand3 *= p
    if n_cand3 <= 10**18:
        score = tau_perp(n_cand3)
        if score > best_score:
            best_score = score
            best_n = n_cand3
            
    # Candidate 4: 2^2 * 3^2 * product of next 12 primes (5 to 43)
    n_cand4 = (2**2) * (3**2)
    for p in primes[2:14]:
        n_cand4 *= p
    if n_cand4 <= 10**18:
        score = tau_perp(n_cand4)
        if score > best_score:
            best_score = score
            best_n = n_cand4
            
    # Candidate 5: 2^5 * 3 * 5 * ... (Check if fits)
    n_cand5 = (2**5) * 3 * 5
    for p in primes[3:13]: # 7 to 41
        n_cand5 *= p
    if n_cand5 <= 10**18:
        score = tau_perp(n_cand5)
        if score > best_score:
            best_score = score
            best_n = n_cand5

    # Candidate 6: 2^3 * 3^2 * 5 * ...
    n_cand6 = (2**3) * (3**2) * 5
    for p in primes[3:13]: # 7 to 41
        n_cand6 *= p
    if n_cand6 <= 10**18:
        score = tau_perp(n_cand6)
        if score > best_score:
            best_score = score
            best_n = n_cand6

    # Check time and return best
    if time.time() > deadline:
        return best_n
        
    return best_n
# EVOLVE-BLOCK-END
