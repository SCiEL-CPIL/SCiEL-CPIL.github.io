import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin, evaluator kills at 90s

    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001

    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(N_LIMIT + 1))
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Precompute powerful parts
    # val[x] will store the powerful part of x
    val = [1] * (N_LIMIT + 1)
    
    # val[1] = 1 (by definition, empty product)
    
    for m in range(2, N_LIMIT + 1):
        x = m
        powerful = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count >= 2:
                powerful *= (p ** count)
        val[m] = powerful

        if time.time() > deadline:
            # Fallback: return best found so far if we run out of time
            # But this loop is very fast, so this is unlikely
            break

    # 3. Find best n
    best_score = -1
    best_n = 10000
    
    # Range [10000, 1000000]
    # We need val[n] and val[n+1], so n goes up to N_MAX
    for n in range(10000, N_MAX + 1):
        score = val[n] * val[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break

    return best_n
