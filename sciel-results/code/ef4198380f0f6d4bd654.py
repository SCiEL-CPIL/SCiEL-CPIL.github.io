"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    # Generate all divisors
    divisors = [1]
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            # Extend divisors
            new_divs = []
            p_power = 1
            for _ in range(count):
                p_power *= p
                for d in divisors:
                    new_divs.append(d * p_power)
            divisors.extend(new_divs)
        p += 1 if p == 2 else 2  # skip even numbers after 2
    if temp > 1:
        new_divs = [d * temp for d in divisors]
        divisors.extend(new_divs)
    
    divisors.sort()
    
    # Count adjacent coprime pairs
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def construct():
    """Construct an integer n <= 10^18 maximizing tau_perp(n)."""
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Strategy: Use squarefree numbers with many prime factors (primorial)
    # and also test prime-power variants that may yield more divisors
    # and different lattice structures with more coprime adjacent pairs.
    
    # First, compute the best primorial-based candidate
    # Product of first 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Generate primorial candidates
    best_n = 1
    best_score = 0
    
    # Try primorial of first k primes for various k
    product = 1
    for k in range(1, len(primes) + 1):
        product *= primes[k - 1]
        if product > 10**18:
            break
        # This is a valid candidate
        score = _tau_perp(product)
        if score > best_score:
            best_score = score
            best_n = product
    
    # Now try prime-power variants
    # The idea: replace some small primes with higher powers to increase divisor count
    # while keeping n <= 10^18
    
    # Try: 2^a * 3^b * 5^c * ... with various exponents
    # Focus on boosting the count of divisors to get more adjacent pairs
    
    # Strategy: start from the 15-prime primorial and try to swap some primes for powers
    # of smaller primes to increase total divisor count
    
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    base_product = 1
    for p in base_primes:
        base_product *= p
    
    # Try various exponent distributions
    # We want to maximize the number of divisors, which is prod(a_i + 1)
    # subject to prod(p_i^a_i) <= 10^18
    
    # Let's try a greedy approach: start with all exponents 1, then increment
    # exponents of the smallest primes as long as we stay under 10^18
    
    # Try different numbers of distinct primes
    for num_primes in range(5, 16):
        if time.time() > deadline - 30:
            break
        
        # Use the smallest num_primes primes
        selected_primes = base_primes[:num_primes]
        
        # Start with all exponents 1
        exponents = [1] * num_primes
        current_product = 1
        for i, p in enumerate(selected_primes):
            current_product *= p ** exponents[i]
        
        # Greedily increase exponents of smallest primes
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(num_primes):
                # Try increasing exponent of prime i
                if current_product * selected_primes[i] <= 10**18:
                    # Check if this improves the score
                    test_product = current_product * selected_primes[i]
                    test_exponents = exponents[:]
                    test_exponents[i] += 1
                    score = _tau_perp(test_product)
                    if score > best_score:
                        best_score = score
                        best_n = test_product
                        current_product = test_product
                        exponents = test_exponents
                        improved = True
                        break
    
    return best_n
# EVOLVE-BLOCK-END
