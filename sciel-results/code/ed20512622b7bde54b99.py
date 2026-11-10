import math

def construct():
    N = 512
    
    # Function to count divisors
    def count_divisors(n):
        count = 0
        # Iterate up to sqrt(n)
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # Find L in [1, N] that maximizes the number of divisors
    best_L = 1
    max_divs = 0
    
    for L in range(1, N + 1):
        d = count_divisors(L)
        if d > max_divs:
            max_divs = d
            best_L = L

    # Generate the set of divisors for best_L
    divisors = []
    for i in range(1, int(math.isqrt(best_L)) + 1):
        if best_L % i == 0:
            divisors.append(i)
            if i != best_L // i:
                divisors.append(best_L // i)
    
    # Ensure non-empty (always true for L>=1)
    if not divisors:
        return [1]
        
    return divisors
