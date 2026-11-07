import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    N = 10_000_000
    
    # Step 1: Sieve for smallest prime factor
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Step 2: Compute sigma using SPF
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    for n in range(2, N + 1):
        if time.time() - start_time > deadline - 5:
            break
        p = spf[n]
        m = n // p
        # n = p^k * m where p doesn't divide m
        # We need to compute the p-part
        if m == 1 or spf[m] != p:
            # p divides n exactly once (in terms of the chain)
            # sigma(n) = sigma(m) * (1 + p)
            sigma[n] = sigma[m] * (1 + p)
        else:
            # p divides m, so we need to find the full power
            # This approach doesn't work directly with SPF
            pass
