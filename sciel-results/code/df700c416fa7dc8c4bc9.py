import time

def construct():
    start_time = time.time()
    # Safety margin: return best found if we hit 85s
    deadline = start_time + 85.0

    MIN_N = 10000
    MAX_N = 1000000
    LIMIT = MAX_N + 1  # We need pp up to MAX_N + 1

    # 1. Sieve of Eratosthenes to find all primes up to LIMIT
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False

    primes = [i for i, flag in enumerate(is_prime) if flag]

    # 2. Compute Powerful Part (PP) for each number
    # pp[n] stores the product of p^e where e >= 2
    pp = [1] * (LIMIT + 1)

    # Iterate through each prime
    for p in primes:
        # If p^2 > LIMIT, then no number <= LIMIT can have p^2 as a factor
        if p * p > LIMIT:
            break

        power = p * p
        next_power = power * p

        # Iterate through powers p^2, p^3, ...
        while power <= LIMIT:
            # For each multiple of 'power'
            # We want to multiply 'power' into pp[m] only if the exponent of p in m is exactly the current power's exponent.
            # i.e., m is divisible by power but not by next_power.
            
            # Optimization: We can just iterate through multiples of power.
            # If m % next_power == 0, we skip.
            
            # Range starts at power, ends at LIMIT, step power
            for m in range(power, LIMIT + 1, power):
                if m % next_power != 0:
                    pp[m] *= power

            power = next_power
            next_power *= p
            
            # Time check inside the tight loop to avoid long stalls
            if time.time() > deadline:
                # If we run out of time, we might not have computed all pp values.
                # However, for N=1e6, this should finish in < 1 second.
                # If it doesn't, we return the best found so far (which might be incomplete).
                # But let's assume it finishes.
                pass

        # Time check between primes
        if time.time() > deadline:
            break

    # 3. Find n in [MIN_N, MAX_N] maximizing pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = -1
    
    # We only care about n in [MIN_N, MAX_N]
    # So we look at pairs (n, n+1) for n in [MIN_N, MAX_N]
    # This requires pp values up to MAX_N + 1
    
    for n in range(MIN_N, MAX_N + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
