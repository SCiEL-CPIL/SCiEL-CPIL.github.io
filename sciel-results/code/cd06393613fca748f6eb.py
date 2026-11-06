import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s wall-clock kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_val = 10**18
    
    best_n = 1
    best_score = -1
    
    def calc_score(exponents):
        """
        Calculate tau_perp given the exponents of the primes in 'primes'.
        """
        # Generate divisors
        divs = [1]
        for i, e in enumerate(exponents):
            if e == 0:
                continue
            p = primes[i]
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def dfs(idx, current_n, exponents, num_divs):
        nonlocal best_n, best_score
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # If current_n > 1, it's a valid candidate
        if current_n > 1:
            # Upper bound on score is num_divs - 1
            if num_divs - 1 > best_score:
                score = calc_score(exponents)
                if score > best_score:
                    best_score = score
                    best_n = current_n
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents from 0 up to max possible
        # e=0 means we don't include this prime
        val = current_n
        e = 0
        while val <= max_val:
            exponents[idx] = e
            # new_num_divs = num_divs * (e + 1)
            new_num_divs = num_divs * (e + 1)
            
            dfs(idx + 1, val, exponents, new_num_divs)
            
            if val > max_val // p:
                break
            val *= p
            e += 1

    # Initialize exponents list
    exponents = [0] * len(primes)
    dfs(0, 1, exponents, 1)
    
    return best_n

if __name__ == "__main__":
    n = construct()
    pass
