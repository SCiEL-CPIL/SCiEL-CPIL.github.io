import math

def construct():
    """
    Maximize the number of adjacent divisor pairs (d_i, d_{i+1}) with gcd(d_i, d_{i+1}) = 1.
    
    Strategy:
    1. Use the largest possible set of distinct small primes (primorial) to maximize the total number of divisors (2^k).
    2. For each prime p in the set, determine its optimal exponent e_p such that the product n = prod p^e_p <= 10^18.
    3. The exponents are chosen to maximize the "gaps" between divisors, which tends to increase the likelihood of coprime adjacency.
       Specifically, larger exponents for smaller primes allow us to "fill" the number range with divisors that have large multiplicative jumps between them.
    
    The optimal exponents are found by a greedy heuristic:
    - Start with e_p = 1 for all primes.
    - While we can increase any exponent without exceeding 10^18, increase the exponent of the smallest prime first, then the next, etc.
      This keeps the base of the power as small as possible, allowing for higher exponents and thus larger gaps in the divisor lattice.
    """
    LIMIT = 10**18
    
    # The 15 smallest primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Initialize exponents to 1
    exponents = [1] * len(primes)
    
    # Calculate current n
    n = 1
    for i, p in enumerate(primes):
        n *= p ** exponents[i]
    
    # Greedily increase exponents
    # We prioritize increasing exponents of smaller primes because:
    # 1. It allows for more total divisors (higher count).
    # 2. It creates larger multiplicative steps for the smallest prime, which often leads to coprime neighbors.
    # We iterate through primes in order, trying to increase their exponent if possible.
    # We do this in a loop until no more increases are possible.
    
    changed = True
    while changed:
        changed = False
        for i in range(len(primes)):
            p = primes[i]
            current_exp = exponents[i]
            # Try to increase exponent by 1
            new_exp = current_exp + 1
            # Check if n * p <= LIMIT
            if n * p <= LIMIT:
                exponents[i] = new_exp
                n *= p
                changed = True
                # We could break here to restart the loop from the beginning,
                # but continuing allows us to potentially increase multiple exponents in one pass.
                # However, to be safe and ensure we always use the smallest primes first,
                # let's restart the loop after any change to ensure we always try the smallest prime first.
                break
    
    # The resulting n is the product of p^e_p
    return n
