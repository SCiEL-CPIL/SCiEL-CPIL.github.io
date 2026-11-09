import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    
    # List of first 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # Find the largest k such that product of first k primes <= limit
    n = 1
    k = 0
    for p in primes:
        if n * p > limit:
            break
        n *= p
        k += 1
        
    # n is now the product of the first k primes
    # This should be the optimal squarefree number
    # We return this n
    
    return n
