import time
from math import isqrt

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0

    N_LIMIT = 10000
    
    # Sieve of Eratosthenes
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N_LIMIT) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n_primes = len(primes)
    
    # DP[j][k] = max length of sequence ending with primes[j], primes[k]
    # Prev[j][k] = index i such that the sequence before primes[j], primes[k] ended with primes[i], primes[j]
    # We only care about j < k.
    # Initialize DP with 0. Base case length is 2.
    DP = [[0] * n_primes for _ in range(n_primes)]
    Prev = [[-1] * n_primes for _ in range(n_primes)]
    
    max_len = 2
    best_j = 0
    best_k = 1
    
    # Iterate over the second-to-last prime index j
    for j in range(n_primes - 1):
        # For a fixed j, we compute DP[j][k] for k > j
        # We maintain a sliding window of valid i's
        # Condition: p_j - p_i <= p_k - p_j  =>  p_i >= 2*p_j - p_k
        # As k increases, 2*p_j - p_k decreases, so the set of valid i's grows.
        
        current_max_val = 0
        current_max_idx = -1
        ptr = j - 1
        
        for k in range(j + 1, n_primes):
            # Check time
            if time.time() > deadline:
                # If we run out of time, we return the best found so far.
                # Since the loops are fast, this is unlikely, but good practice.
                break
            
            target = 2 * primes[j] - primes[k]
            
            # Update current_max with new valid i's
            # We need primes[ptr] >= target
            while ptr >= 0 and primes[ptr] >= target:
                if DP[ptr][j] > current_max_val:
                    current_max_val = DP[ptr][j]
                    current_max_idx = ptr
                ptr -= 1
            
            if current_max_val > 0:
                DP[j][k] = current_max_val + 1
                Prev[j][k] = current_max_idx
            else:
                DP[j][k] = 2
                Prev[j][k] = -1
            
            if DP[j][k] > max_len:
                max_len = DP[j][k]
                best_j = j
                best_k = k
        
        # If time is exceeded, we might break outer loop too, but let's just let it finish 
        # as it's O(N^2) and N is small.
        if time.time() > deadline:
            break

    # Backtrack to get the sequence
    seq = []
    if max_len == 2:
        seq = [primes[best_j], primes[best_k]]
    else:
        j = best_j
        k = best_k
        seq.append(primes[k])
        seq.append(primes[j])
        
        while Prev[j][k] != -1:
            i = Prev[j][k]
            seq.append(primes[i])
            # The previous state was (i, j)
            k = j
            j = i
        
        seq.reverse()
        
    return seq

# Run construct to verify
# print(construct())
