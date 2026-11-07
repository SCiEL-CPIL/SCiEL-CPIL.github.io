"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = []
    
    for p in primes:
        # Find max exponent k such that p^k <= 40
        max_exp = 0
        current_power = 1
        while current_power * p <= 40:
            current_power *= p
            max_exp += 1
            
        # We want to select a subset of exponents {1, ..., max_exp}
        # such that all subset sums are distinct.
        # The set {1, 2, 4, ..., 2^(m-1)} has this property.
        # We greedily pick exponents 1, 2, 4, ... as long as they are <= max_exp.
        # Note: This maximizes the number of selected exponents for a given max_exp.
        
        selected_exponents = []
        next_power_of_2 = 1
        while next_power_of_2 <= max_exp:
            selected_exponents.append(next_power_of_2)
            next_power_of_2 *= 2
            
        # Generate the numbers p^e for e in selected_exponents
        for e in selected_exponents:
            result.append(p ** e)
            
    return result
# EVOLVE-BLOCK-END
