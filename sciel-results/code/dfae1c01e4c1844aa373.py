import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 1000001
    # powerful[n] will store the powerful part of n
    # Using int64 to avoid overflow, though max product might exceed int32
    # Max powerful part is n itself, so max value is 1e6.
    # Product of two is 1e12, which fits in int64.
    powerful = np.ones(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to find primes up to sqrt(N)
    # We only need primes p such that p^2 <= N, i.e., p <= sqrt(N)
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # For each prime, update the powerful array
    for p in primes:
        # Find max power p^K <= N
        power = p * p
        K = 2
        while power * p <= N:
            power *= p
            K += 1
        
        # Iterate from highest power down to 2
        # We need to handle p^K, p^(K-1), ..., p^2
        # Let's generate the powers
        powers = []
        val = p
        for _ in range(K):
            val *= p
            if val > N:
                break
            powers.append(val)
        
        # powers contains [p^1, p^2, ..., p^K]
        # We care about indices 1 to K-1 (which correspond to p^2 to p^K)
        # Iterate backwards
        for j in range(len(powers) - 1, 0, -1):
            q = powers[j]       # p^(j+1) ? No, powers[0]=p, powers[1]=p^2.
            # If j=1, q=p^2. If j=len-1, q=p^K.
            # The exponent is j+1.
            # We want to mark multiples of q.
            # Check if they are multiples of next power p^(j+2).
            # If j < len(powers)-1, next_power = powers[j+1].
            # Else next_power = None (infinity).
            
            idx = np.arange(q, N + 1, q)
            
            if j < len(powers) - 1:
                next_q = powers[j+1]
                # Mask out multiples of next_q
                # idx % next_q != 0
                mask = (idx % next_q) != 0
                idx = idx[mask]
            
            # Multiply powerful[idx] by q (which is p^(exponent))
            powerful[idx] *= q
            
        # Check deadline
        if time.time() > deadline:
            break

    # Calculate scores for n in [10000, 1000000]
    # n goes from 10000 to 1000000 inclusive
    # We need powerful[n] and powerful[n+1]
    # powerful array is 0..N
    # n=10000 -> powerful[10000] * powerful[10001]
    # n=1000000 -> powerful[1000000] * powerful[1000001]
    
    # Slice powerful from 10000 to 1000000 (inclusive)
    # This is powerful[10000 : 1000001]
    # Next values are powerful[10001 : 1000002]
    
    # Ensure N is large enough
    # N = 1000001, so index 1000001 is valid.
    
    left = powerful[10000:1000001]
    right = powerful[10001:1000002]
    
    scores = left * right
    
    if len(scores) == 0:
        return 10000
        
    best_idx = np.argmax(scores)
    best_n = 10000 + best_idx
    
    return int(best_n)
