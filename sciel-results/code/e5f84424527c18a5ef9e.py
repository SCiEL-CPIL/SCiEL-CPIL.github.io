import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing tau_perp(n).
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin for 150s kill
    MAX_VAL = 10**18
    
    # First 15 primes are sufficient for n <= 10^18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 0
    best_n = 2
    
    def compute_score(primes_list, exponents_list):
        """
        Computes tau_perp for a given factorization.
        """
        # Generate all divisors
        divisors = [1]
        for p, e in zip(primes_list, exponents_list):
            new_divisors = []
            power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divisors.append(d * power)
                power *= p
            divisors = new_divisors
        
        # Sort divisors
        divisors.sort()
        
        # Count coprime adjacent pairs
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count

    def search(idx, max_exp, current_n, primes_used, exponents_used):
        nonlocal best_score, best_n
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try all exponents from max_exp down to 1
        # We can also stop if current_n * p > MAX_VAL
        for e in range(max_exp, 0, -1):
            new_n = current_n * (p ** e)
            if new_n > MAX_VAL:
                continue
                
            # Update state
            primes_used.append(p)
            exponents_used.append(e)
            
            # Evaluate this candidate
            # Optimization: Only compute score if this is a "leaf" or if we want to check intermediate?
            # Actually, we should check every valid n.
            # But computing score is expensive. 
            # However, the number of candidates is small enough.
            
            score = compute_score(primes_used, exponents_used)
            if score > best_score:
                best_score = score
                best_n = new_n
                
            # Recurse to next prime
            # Next max exponent is e (non-increasing exponents)
            search(idx + 1, e, new_n, primes_used, exponents_used)
            
            # Backtrack
            primes_used.pop()
            exponents_used.pop()
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start search
    # Max exponent for 2 is floor(log2(10^18)) = 59
    search(0, 60, 1, [], [])
    
    return best_n

if __name__ == "__main__":
    n = construct()
    print(n)
