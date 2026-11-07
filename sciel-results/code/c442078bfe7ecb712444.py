import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Primes up to 40
    primes = []
    for i in range(2, 41):
        if all(i % d for d in range(2, isqrt(i) + 1)):
            primes.append(i)
            
    # The maximum number of multiplicatively independent integers 
    # in [2, N] is the number of primes <= N.
    # This is because the exponent vectors live in a vector space 
    # with dimension equal to the number of primes <= N.
    # The prime numbers themselves correspond to the standard basis vectors.
    
    # To be robust and "compute" the answer rather than hardcoding logic:
    # We can verify that the primes are independent (they are) 
    # and that no more can be added (dimension limit).
    
    # For N=40, primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Count: 12.
    
    # We return the list of primes.
    return primes
