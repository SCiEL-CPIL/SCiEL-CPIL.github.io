import time

def construct():
    """
    Returns a list of 16 distinct integers in 2..40 such that all 2^16 subset products are distinct.
    
    The set is constructed by selecting prime powers where the exponents for each prime
    form a set with distinct subset sums.
    
    Primes and selected powers:
    - p=2: exponents {1, 2, 4} -> numbers 2, 4, 16
    - p=3: exponents {1, 2} -> numbers 3, 9
    - p=5: exponents {1, 2} -> numbers 5, 25
    - p=7: exponent {1} -> number 7
    - p=11: exponent {1} -> number 11
    - p=13: exponent {1} -> number 13
    - p=17: exponent {1} -> number 17
    - p=19: exponent {1} -> number 19
    - p=23: exponent {1} -> number 23
    - p=29: exponent {1} -> number 29
    - p=31: exponent {1} -> number 31
    - p=37: exponent {1} -> number 37
    
    Total count: 3 + 2 + 2 + 1*8 = 16.
    This is the maximum possible size because:
    - For p=2, exponents 1..5 cannot support 4 numbers (max range 15 < 16 required distinct sums).
    - For p=3, exponents 1..3 cannot support 3 numbers (max range 7 < 8 required distinct sums).
    - Other primes support at most 2 or 1 numbers.
    - Mixing primes reduces efficiency as it consumes capacity from multiple primes for a single number.
    """
    # Hardcoded optimal set
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure types are strictly Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
