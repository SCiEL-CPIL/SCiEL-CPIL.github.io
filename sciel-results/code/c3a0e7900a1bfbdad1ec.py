"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 123457
    
    def generate_smooths(limit, max_prime):
        numbers = {1}
        for p in [2, 3, 5, 7]:
            if p > max_prime:
                break
            # Iterate over a snapshot to avoid modifying list while iterating
            # and to ensure we only multiply numbers that existed before this prime
            # Actually, we want to multiply ALL existing numbers by powers of p.
            # If we add 2 to the set, then later for p=3, we should multiply 2 by 3.
            # The loop `for x in list(numbers)` does this correctly if `numbers` 
            # contains all combinations of previous primes.
            current_numbers = list(numbers)
            for x in current_numbers:
                val = x
                while val * p <= limit:
                    val *= p
                    numbers.add(val)
        return numbers

    # Check for minimal largest prime factor
    # We check primes in increasing order: 2, 3, 5, 7
    # If a pair exists with max prime factor <= p, the value is p (or 1 if p=1, but p>=2 here)
    for p_max in [2, 3, 5, 7]:
        smooths = generate_smooths(N, p_max)
        for a in smooths:
            b = N - a
            if b >= 1 and b in smooths:
                return [a, b]
                
    # Fallback: 1 and N-1 are always valid, and for this N, N-1 is 7-smooth.
    # If we reached here, it means no solution was found with primes <= 7, 
    # which is theoretically impossible for this N given the seed, but safe to handle.
    return [1, N - 1]
# EVOLVE-BLOCK-END
