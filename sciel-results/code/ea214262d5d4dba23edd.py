import time
import sys

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # n+i where i<=24, n<=1000000
    
    # Precompute the extracted factor for each number up to max_val
    # extracted[m] = product of p^a for p in small_primes where p^a divides m
    extracted = [1] * (max_val + 1)
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            power = 1
            val = multiple
            while val % p == 0:
                val //= p
                power *= p
            extracted[multiple] *= power
    
    # Now search for n in [1000, max_n] minimizing distinct extracted values in [n+1, ..., n+24]
    best_score = float('inf')
    best_n = 1000
    
    # Use a sliding window approach with a frequency count
    # For efficiency, iterate and maintain a count of distinct values
    
    start = 1000
    end = max_n
    
    # First pass: coarse search with stride
    stride = 1
    for n in range(start, end + 1, stride):
        if time.time() > deadline:
            break
        # Count distinct extracted values in n+1 .. n+24
        vals = extracted[n+1:n+25]
        distinct = len(set(vals))
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 1:
                break  # can't do better than 1
    
    return best_n
