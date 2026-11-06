import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 1000000
    MAX_VAL = N + 1

    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = np.where(is_prime)[0].tolist()

    # Initialize powerful part array as float64 for NumPy operations
    P = np.ones(MAX_VAL + 1, dtype=np.float64)

    # Compute powerful parts using NumPy vectorized operations
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            next_power = power * p
            if next_power > MAX_VAL:
                # For all multiples of 'power' in range, the exponent is exactly this
                indices = np.arange(power, MAX_VAL + 1, power)
                P[indices] *= power
                break
            else:
                # Multiply by 'power' for multiples of 'power' that are NOT multiples of 'next_power'
                indices = np.arange(power, MAX_VAL + 1, power)
                mask = (indices % next_power) != 0
                P[indices[mask]] *= power
            power = next_power

        # Deadline check
        if time.time() > deadline:
            break

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    # Use NumPy for fast max search
    products = P[10000:N+1] * P[10001:N+2]
    best_idx = np.argmax(products)
    best_n = 10000 + best_idx
    best_val = products[best_idx]

    return int(best_n)
