import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000

    # Initialize powerful part array with 1s
    # Using int64 to be safe, though int32 would likely work
    pp = np.ones(MAXN + 1, dtype=np.int64)

    # Sieve for primes up to MAXN
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]

    # Calculate powerful parts using optimized sieve
    for p in primes:
        if p * p > MAXN:
            break

        p_power = p * p
        # First power (k=2) multiplies by p^2
        # Subsequent powers (k>2) multiply by p
        while p_power <= MAXN:
            # Multiply all multiples of p_power by the appropriate factor
            # For k=2 (first iteration), factor is p^2
            # For k>2, factor is p
            if p_power == p * p:
                pp[p_power::p_power] *= p_power
            else:
                pp[p_power::p_power] *= p
            
            p_power *= p

            if elapsed() > DEADLINE:
                # If we time out here, we return a fallback
                # But this loop is very fast for most primes
                return 10000

    if elapsed() > DEADLINE:
        return 10000

    # Scan for the best pair
    start = 10000
    # n ranges from start to MAXN-1 (so that n+1 <= MAXN)
    # We want to maximize pp[n] * pp[n+1]
    # Slice pp[start:MAXN] corresponds to n=start ... MAXN-1
    # Slice pp[start+1:MAXN+1] corresponds to n+1=start+1 ... MAXN
    
    products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    # Find the index of the max product
    max_idx = np.argmax(products)
    best_n = start + max_idx

    return int(best_n)
