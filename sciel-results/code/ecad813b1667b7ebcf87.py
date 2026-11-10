import time
import math

def construct():
    start = time.time()
    DEADLINE = 60.0  # safety margin before the 90s kill

    N = 1000000          # max allowed n
    NMAX = N + 1         # we need lpf for values up to 1000001

    # --- Sieve primes up to NMAX ---
    # Using a bytearray for the sieve is memory and time efficient
    is_prime = bytearray(b'\x01') * (NMAX + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    limit = math.isqrt(NMAX)
    for i in range(2, limit + 1):
        if is_prime[i]:
            # Mark multiples of i starting from i*i
            start_idx = i * i
            is_prime[start_idx : NMAX + 1 : i] = bytearray(len(is_prime[start_idx : NMAX + 1 : i]))

    # --- Largest prime factor for every integer up to NMAX ---
    # Initialize lpf array
    lpf = [0] * (NMAX + 1)
    
    # Stamp largest prime factor
    # Iterate through primes in increasing order.
    # The last prime that divides a number m will be its largest prime factor.
    for p in range(2, NMAX + 1):
        if is_prime[p]:
            # Optimization: only iterate if p is prime
            for m in range(p, NMAX + 1, p):
                lpf[m] = p

    # --- Scan the whole window for the global minimum of max(lpf[n], lpf[n+1]) ---
    best_n = 10000
    # Initialize with the first valid pair
    val_10000 = lpf[10000]
    val_10001 = lpf[10001]
    best_val = val_10000 if val_10000 > val_10001 else val_10001

    # Scan remaining n
    for n in range(10001, N + 1):
        v1 = lpf[n]
        v2 = lpf[n + 1]
        val = v1 if v1 > v2 else v2
        
        if val < best_val:
            best_val = val
            best_n = n
        
        # Periodic deadline guard: bail out with the best found so far
        if (n & 0xFFFF) == 0 and time.time() > DEADLINE - 2.0:
            break

    return best_n
