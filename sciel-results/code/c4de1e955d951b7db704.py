import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Sieve for primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    num_primes = len(primes)

    MAX_L = 100000  # We'll try to cover up to this (capped by score)

    # Use NumPy for fast coverage
    covered = np.zeros(MAX_L + 2, dtype=np.uint8)
    classes = []
    used_primes = set()
    current_sum_num = 0  # numerator of sum
    current_sum_den = 1  # denominator of sum
    # Track budget as Fraction for exactness
    budget = Fraction(1, 1)
    current_sum_frac = Fraction(0, 1)

    # Precompute prime costs as fractions
    prime_costs = {p: Fraction(1, p) for p in primes}

    best_L = 0
    n = 1

    while n <= MAX_L:
        if time.time() > deadline:
            break

        if covered[n]:
            best_L = n
            n += 1
            continue

        # n is not covered, need to add a class
        # Try each unused prime, pick the one that maximizes new coverage
        best_p = None
        best_new_cov = -1
        best_r = None

        for p in primes:
            if p in used_primes:
                continue
            cost = prime_costs[p]
            if current_sum_frac + cost > budget:
                continue

            r = n % p
            # The first number >= n covered by this class
            if r >= n:
                first_x = r
            else:
                first_x = r + ((n - r + p - 1) // p) * p

            # Count new covers in [first_x, MAX_L] with step p
            # Use NumPy for speed
            indices = np.arange(first_x, MAX_L + 1, p)
            new_cov = int(np.sum(covered[indices] == 0))

            # Prefer primes that cover more new numbers
            # Tie-break: prefer larger p (cheaper cost)
            if new_cov > best_new_cov or (new_cov == best_new_cov and (best_p is None or p > best_p)):
                best_new_cov = new_cov
                best_p = p
                best_r = r

        if best_p is None:
            # Can't cover n, stop
            break

        # Add the class
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        current_sum_frac += prime_costs[best_p]

        # Update covered using NumPy
        p = best_p
        r = best_r
        if r >= 1:
            indices = np.arange(r, MAX_L + 1, p)
            covered[indices] = 1
        else:
            # r == 0, so cover p, 2p, 3p, ...
            indices = np.arange(p, MAX_L + 1, p)
            covered[indices] = 1

        best_L = n
        n += 1

    return classes
