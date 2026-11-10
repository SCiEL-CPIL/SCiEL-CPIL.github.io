import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(200)
    
    def tau_perp(n, prime_factors, exponents):
        divs = [1]
        for p, e in zip(prime_factors, exponents):
            new_divs = []
            pow_p = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * pow_p)
                pow_p *= p
            divs = new_divs
        divs.sort()
        score = 0
        gcd = math.gcd
        # Optimization: Unroll the loop or use local vars for speed
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    def calc_n(prime_factors, exponents):
        n = 1
        for p, e in zip(prime_factors, exponents):
            n *= p ** e
        return n

    best_n = 2
    best_score = 1

    # Targeted Search Strategy:
    # Based on knowledge, increasing exponents of small primes (especially 2) is beneficial.
    # We will search over the number of primes K (10 to 15) and the exponents of the first few primes.
    
    max_k = 15
    
    for k in range(10, max_k + 1):
        if time.time() > deadline - 10.0:
            return best_n
            
        p_set = primes[:k]
        base_exponents = [1] * k
        base_n = 1
        for p in p_set:
            base_n *= p
            
        if base_n > 10**18:
            continue
            
        # Search space:
        # e_0 (for prime 2): 1 to max
        # e_1 (for prime 3): 1 to max
        # e_2 (for prime 5): 1 to max (optional, keep small)
        # Others: 1
        
        # Calculate max exponents for first few primes
        # We iterate e0, e1, e2 and check if n <= 10^18
        
        max_e0 = 1
        while base_n * (2 ** max_e0) <= 10**18:
            max_e0 += 1
        max_e0 -= 1 # max valid is where it becomes >, so step back? No, loop condition handles it.
        # Actually, let's just loop and break.
        
        # To be efficient, we nest loops but break early.
        
        # We will try a grid of exponents for p0, p1, p2.
        # p0 = 2, p1 = 3, p2 = 5
        
        for e0 in range(1, max_e0 + 1):
            if time.time() > deadline - 5.0:
                return best_n
            n_e0 = base_n * (2 ** (e0 - 1))
            if n_e0 > 10**18:
                break
                
            # Max e1 given e0
            max_e1 = 1
            while n_e0 * (3 ** max_e1) <= 10**18:
                max_e1 += 1
            max_e1 -= 1
            
            for e1 in range(1, max_e1 + 1):
                if time.time() > deadline - 5.0:
                    return best_n
                n_e01 = n_e0 * (3 ** (e1 - 1))
                if n_e01 > 10**18:
                    break
                    
                # Max e2 given e0, e1
                max_e2 = 1
                while n_e01 * (5 ** max_e2) <= 10**18:
                    max_e2 += 1
                max_e2 -= 1
                
                for e2 in range(1, max_e2 + 1):
                    if time.time() > deadline - 2.0:
                        return best_n
                    n_candidate = n_e01 * (5 ** (e2 - 1))
                    if n_candidate > 10**18:
                        break
                    
                    # Construct exponents list
                    exponents = [1] * k
                    exponents[0] = e0
                    exponents[1] = e1
                    exponents[2] = e2
                    
                    score = tau_perp(n_candidate, p_set, exponents)
                    
                    if score > best_score:
                        best_score = score
                        best_n = n_candidate

    return best_n
