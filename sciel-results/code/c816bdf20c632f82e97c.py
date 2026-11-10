"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find an integer n <= 10^18 that maximizes tau_perp(n), the number of
    adjacent pairs of divisors with gcd 1.
    
    Strategy:
    1. The seed solution (product of first 8 primes) has a score of ~21.92.
    2. We know that using more prime factors (up to the largest primorial <= 10^18)
       increases the score. The largest primorial <= 10^18 is the product of the
       first 15 primes, which is 614889782588491410.
    3. However, we can potentially do better by using prime powers. 
       For example, replacing a large prime factor with a higher power of a smaller
       prime might allow more divisors and more opportunities for coprime adjacent pairs.
       
    Let's analyze the structure:
    If n = p1^a1 * p2^a2 * ... * pk^ak, the divisors are all products p1^b1 * ... * pk^bk 
    where 0 <= bi <= ai.
    
    The key insight from the observed data is that "prime power structures (e.g., 2^4 * prod odd primes) 
    which increase divisor count and lattice density, allowing more adjacent coprime pairs than squarefree numbers".
    
    The product of the first 15 primes is ~6.1 * 10^17. We have a budget of 10^18.
    10^18 / 6.1 * 10^17 ≈ 1.63.
    So we can multiply by a small factor, or replace some primes with higher powers.
    
    Let's try to optimize the exponents.
    Let n = 2^a * 3^b * 5^c * ... * p_k^e_k.
    
    We want to maximize the number of adjacent coprime pairs.
    A pair (d, d') is coprime if they share no prime factors.
    In the sorted list of divisors, d and d' are adjacent.
    
    Consider the simplest case: n is square-free with k primes.
    The number of divisors is 2^k.
    The score for k=15 is the baseline.
    
    What if we use n = 2^a * 3 * 5 * ... * p_k?
    The number of divisors is (a+1) * 2^(k-1).
    
    Let's write a helper to compute the score for a given n.
    Since n can be up to 10^18, the number of divisors can be large (up to ~100,000 for highly composite numbers).
    Generating all divisors, sorting them, and checking gcd for adjacent pairs is feasible within 150s.
    
    We will perform a search over possible factorizations.
    Since the number of distinct prime factors is limited (max 15), we can iterate over possible exponent vectors.
    
    However, a full search over exponents for 15 primes might be too large.
    Let's restrict our search to the first 15 primes, and try small exponents for the smaller primes.
    
    Actually, the optimal structure is likely to involve small primes with higher exponents.
    Let's try to maximize the number of divisors first, as more divisors generally means more adjacent pairs.
    But we specifically want *coprime* adjacent pairs.
    
    Let's compute the score for the known best candidate (primorial of 15) first.
    Then, we'll try variations.
    
    Primes:
    p = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    Base: n0 = prod(p) = 614889782588491410
    Score(n0) = ?
    
    Let's write a function to compute the score.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_score(n):
        """Compute tau_perp(n)"""
        if n < 2:
            return 0
        # Factorize n
        # Since n is composed of small primes, we can trial divide
        factors = {}
        temp_n = n
        for p in primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors[p] = count
        if temp_n > 1:
            factors[temp_n] = 1
            
        # Generate all divisors
        divisors = [1]
        for p, e in factors.items():
            new_divisors = []
            for d in divisors:
                pe = 1
                for _ in range(e + 1):
                    new_divisors.append(d * pe)
                    pe *= p
            divisors = new_divisors
            
        divisors.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score
    
    # Initial best: primorial of 15
    best_n = 1
    for p in primes:
        best_n *= p
    best_score = get_score(best_n)
    
    # Now, let's try to improve by adjusting exponents.
    # We have budget 10^18.
    # Current n0 is ~6.1e17.
    # We can multiply by factors of 2, 3, 5, etc., as long as n <= 10^18.
    
    # Let's do a DFS over exponents for the first few primes.
    # Since 2, 3, 5 are small, their exponents can be larger.
    # The remaining primes can likely stay at exponent 1.
    
    # Let's try to optimize exponents for 2, 3, 5, 7, 11, 13.
    # Assume primes from 17 onwards have exponent 1.
    
    # Let's use a recursive search.
    # State: index of prime, current product, current score (or just compute at the end).
    # Since computing score is expensive, we should be selective.
    
    # Heuristic: More divisors usually means more coprime adjacent pairs.
    # Let's first find the number with the maximum number of divisors <= 10^18.
    # Then check its score.
    # Then check neighbors in the lattice of factorizations.
    
    # Actually, let's just try a few specific candidates:
    candidates = []
    
    # 1. Primorial 15
    n1 = 1
    for p in primes:
        n1 *= p
    candidates.append(n1)
    
    # 2. Try increasing exponent of 2
    # n1 * 2 = 1.23e18 > 10^18. So we can't just multiply by 2.
    # We need to replace some primes or reduce exponents elsewhere? No, exponents are 1.
    # We can replace a large prime with a power of a smaller one?
    # E.g., replace 47 with 2^k?
    # 47 / 2 = 23.5. So 2^5 = 32 < 47. 2^6 = 64 > 47.
    # So n = (n1 / 47) * 2^5 is valid?
    # n1 / 47 = 1.3e16. * 32 = 4.16e17. Valid.
    # Does this increase score?
    # Divisors of n1: 2^15 = 32768.
    # Divisors of new n: 2^5 * 3 * 5 * ... * 43.
    # Exponents: 2^5, others 1.
    # Num divisors: (5+1) * 2^14 = 6 * 16384 = 98304.
    # Much more divisors! Likely higher score.
    
    # Let's systematically try replacing the largest primes with powers of 2.
    
    # General strategy:
    # Start with n = product of first k primes.
    # Try to "upgrade" small primes to higher powers by "downgrading" or removing large primes.
    
    # Let's write a function to generate candidates based on exponent vectors for the first M primes,
    # and assume the rest are 1 or 0.
    
    # Given the time limit, we'll do a limited search.
    
    # Let's try all combinations of exponents for 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    # such that the product <= 10^18.
    # This is too large a space.
    
    # Better:
    # 1. Compute score for n1 (primorial 15).
    # 2. Try replacing the largest prime p_k with p_j^e for j < k.
    # 3. Try replacing two largest primes with powers of smaller ones.
    
    # Let's implement a greedy local search.
    # Start with best_n.
    # Try to change one prime's exponent.
    # If n = p1^a1 * ... * pk^ak.
    # Try to increase a_i by 1 (multiply by p_i) and decrease a_j by 1 (divide by p_j) to stay <= 10^18.
    # Or just try to multiply by p_i if possible.
    
    # Let's just enumerate a few promising structures.
    
    # Structure A: 2^a * 3^b * 5 * 7 * ... * p_k
    # Structure B: 2^a * 3 * 5 * 7 * ... * p_k
    
    # Let's try to maximize the number of divisors first.
    # The number with the max divisors <= 10^18 is known.
    # It is 897612484786617600 (or similar).
    # But we want coprime adjacent pairs, not just total divisors.
    
    # Let's try a specific candidate:
    # n = 2^5 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    # Let's check if this is <= 10^18.
    
    # We'll do a DFS to find good candidates.
    
    def dfs(idx, current_n, current_factors):
        """
        idx: index in primes list
        current_n: product so far
        current_factors: dict of prime -> exponent
        """
        nonlocal best_n, best_score
        
        if time.time() > deadline:
            return
            
        # Prune if current_n > 10^18
        if current_n > 10**18:
            return
            
        # If we've considered all primes, evaluate
        if idx == len(primes):
            score = get_score(current_n)
            if score > best_score:
                best_score = score
                best_n = current_n
            return
            
        p = primes[idx]
        
        # Try exponents from 0 up to max possible
        # Max exponent: current_n * p^e <= 10^18
        # p^e <= 10^18 / current_n
        max_e = 0
        if current_n > 0:
            limit = 10**18 // current_n
            if limit >= 1:
                e = 0
                val = 1
                while val * p <= limit:
                    val *= p
                    e += 1
                max_e = e
        
        # We don't want to try all exponents up to max_e for all primes.
        # Let's limit the search depth or branching.
        # For the first few primes, try all exponents.
        # For later primes, try only 0 and 1 (or up to 2).
        
        if idx < 6:
            # Small primes: try all exponents
            for e in range(max_e + 1):
                current_factors[p] = e
                dfs(idx + 1, current_n * (p ** e), current_factors)
        else:
            # Larger primes: try 0 and 1 (and maybe 2 if small enough)
            for e in range(min(max_e, 2) + 1):
                current_factors[p] = e
                dfs(idx + 1, current_n * (p ** e), current_factors)
                
    # This DFS might be too slow or explore too many branches.
    # Let's try a simpler heuristic:
    # 1. Start with primorial 15.
    # 2. Try to replace the largest prime with a power of 2.
    # 3. Try to replace the two largest primes with powers of 2 and 3.
    
    # Let's just try a few specific candidates manually generated.
    
    # Candidate 1: Primorial 15
    n = 1
    for p in primes:
        n *= p
    candidates.append(n)
    
    # Candidate 2: Replace 47 with 2^5
    # n = (primorial15 / 47) * 32
    n = (1)
    for p in primes:
        if p != 47:
            n *= p
    n *= 32
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 3: Replace 47 and 43 with 2^4 * 3^2
    n = 1
    for p in primes:
        if p not in [47, 43]:
            n *= p
    n *= 16 * 9
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 4: Max divisors number (approx)
    # 2^a * 3^b * 5^c ...
    # Let's try 2^5 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    n = 2**5 * 3**3 * 5**2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 5: 2^4 * 3^4 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    n = 2**4 * 3**4 * 5**2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 6: 2^6 * 3^3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    n = 2**6 * 3**3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 7: 2^7 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    n = 2**7 * 3**2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 8: 2^8 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    n = 2**8 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 9: 2^9 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    n = 2**9 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    if n <= 10**18:
        candidates.append(n)
        
    # Candidate 10: 2^10 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    n = 2**10 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    if n <= 10**18:
        candidates.append(n)
        
    # Evaluate all candidates
    for n in candidates:
        if n < 2 or n > 10**18:
            continue
        score = get_score(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
