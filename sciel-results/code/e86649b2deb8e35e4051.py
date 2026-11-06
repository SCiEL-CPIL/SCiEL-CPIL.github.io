"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit (evaluator kills at 90s, but be safe)

    # We need to evaluate n in [10000, 1000000], so we need powerful part up to 1000001
    MAXN = 1_000_001

    # 1. Sieve of Eratosthenes for primes up to MAXN
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = np.where(is_prime)[0]

    # 2. Compute h(x) = product of p where v_p(x) == 1 (the square-free part with exponent exactly 1)
    # Then powerful_part(x) = x // h[x]
    # 
    # Strategy: 
    #   Start with h = 1 for all x.
    #   For each prime p:
    #     Multiply h[m] by p for all multiples m of p (adds p to square-free part)
    #     Divide h[m] by p for all multiples m of p^2 (removes p if exponent >= 2)
    #   This gives h[x] = product of primes p where v_p(x) == 1
    #   Then powerful_part(x) = x // h[x] = product of p^e where e >= 2

    h = np.ones(MAXN + 1, dtype=np.int64)

    for p in primes:
        p = int(p)
        # Multiply by p for all multiples of p
        h[p::p] *= p
        
        # Divide by p for all multiples of p^2
        p2 = p * p
        if p2 <= MAXN:
            h[p2::p2] //= p

    if time.time() - t0 > DEADLINE:
        return 10000

    # 3. Compute P[x] = powerful_part(x) = x // h[x]
    x_vals = np.arange(MAXN + 1, dtype=np.int64)
    P = x_vals // h

    if time.time() - t0 > DEADLINE:
        return 10000

    # 4. Maximize P[n] * P[n+1] for n in [10000, 1000000]
    start = 10000
    end = 1000000

    # We need P[start] through P[end+1]
    P_slice = P[start:end+2]  # Length = end - start + 2

    # products[i] = P_slice[i] * P_slice[i+1] = P[n] * P[n+1] where n = start + i
    products = P_slice[:-1] * P_slice[1:]

    # Find index of max
    idx = int(np.argmax(products))
    best_n = start + idx

    # Ensure best_n is in valid range
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000

    return best_n
# EVOLVE-BLOCK-END
