import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 80  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000

    # Initialize powerful part array with 1s
    pp = np.ones(MAXN + 2, dtype=np.int64)

    # Sieve for primes up to MAXN
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]

    # Calculate powerful parts using sieve
    for p in primes:
        if p * p > MAXN:
            break

        p_j = p * p
        while p_j <= MAXN:
            p_next = p_j * p

            # Get indices of multiples of p_j
            indices = np.arange(p_j, MAXN + 1, p_j)

            if p_next > MAXN:
                # All multiples of p_j have v_p(m) >= j (no higher power to exclude)
                pp[indices] *= p_j
            else:
                # Filter out multiples of p_next (those have v_p(m) > j)
                mask = indices % p_next != 0
                pp[indices[mask]] *= p_j

            p_j = p_next

            if elapsed() > DEADLINE:
                break

    if elapsed() > DEADLINE:
        return 10000

    # Scan for the best pair
    start = 10000
    # n ranges from start to MAXN-1 (so that n+1 <= MAXN)
    # pp[n] and pp[n+1] are valid
    products = pp[start:MAXN] * pp[start + 1:MAXN + 1]

    # Find the index of the max product
    max_idx = np.argmax(products)
    best_n = start + max_idx

    return int(best_n)
