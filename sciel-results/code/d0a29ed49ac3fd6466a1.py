import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 1000000
    MAX_VAL = N + 1

    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, v in enumerate(is_prime) if v]

    # Initialize powerful part array
    P = [1] * (MAX_VAL + 1)

    # Compute powerful parts using segmented sieve over primes
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            next_power = power * p
            if next_power > MAX_VAL:
                # For all multiples of 'power' in range, the exponent is exactly this
                for m in range(power, MAX_VAL + 1, power):
                    P[m] *= power
                break
            else:
                # Multiply by 'power' for multiples of 'power' that are NOT multiples of 'next_power'
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        P[m] *= power
            power = next_power

        # Deadline check
        if time.time() > deadline:
            break

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    best_n = 10000
    best_val = 0

    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

        if time.time() > deadline:
            break

    return best_n
