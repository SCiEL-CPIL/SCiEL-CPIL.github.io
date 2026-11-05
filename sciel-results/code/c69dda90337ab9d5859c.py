import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 1000000
    
    # Sieve of Eratosthenes to get primes up to sqrt(N)
    sqrt_N = int(math.isqrt(N))
    is_prime = bytearray(b'\x01') * (sqrt_N + 1)
    is_prime[0] = 0
    if sqrt_N >= 1:
        is_prime[1] = 0
    for i in range(2, int(math.isqrt(sqrt_N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, sqrt_N + 1) if is_prime[i]]
    
    # Use NumPy array for faster vectorized updates
    # P[n] stores the powerful part of n
    # Max P[n] <= N = 10^6, so int64 is safe
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        pk = p2
        k = 2
        while pk <= N:
            next_pk = pk * p
            if next_pk > N:
                # This is the last power for this prime
                # Indices are multiples of pk
                # We need to exclude multiples of next_pk (which don't exist <= N if next_pk > N? No, next_pk > N means no multiples of next_pk in range)
                # Wait, if next_pk > N, then ALL multiples of pk have v_p(m) = k exactly?
                # No. m = j * pk. v_p(m) = k + v_p(j).
                # If next_pk > N, then m < next_pk implies v_p(j) must be 0?
                # m = j * pk. If j is divisible by p, m is divisible by p * pk = next_pk.
                # If next_pk > N, then no m <= N is divisible by next_pk.
                # So NO m <= N has v_p(m) >= k+1.
                # So ALL multiples of pk have v_p(m) = k.
                # So we can update ALL multiples of pk.
                idx = np.arange(pk, N + 1, pk)
                P[idx] *= pk
            else:
                # We need multiples of pk that are NOT multiples of next_pk
                # idx_all = np.arange(pk, N + 1, pk)
                # mask = (idx_all % next_pk) != 0
                # idx = idx_all[mask]
                # P[idx] *= pk
                
                # Optimization:
                # Multiples of pk: pk, 2pk, 3pk, ...
                # Multiples of next_pk: next_pk, 2next_pk, ... = p*pk, 2p*pk, ...
                # So we want indices j in [1, N//pk] such that j % p != 0.
                # This is equivalent to taking all indices and removing every p-th one.
                
                limit = N // pk
                # Create array of all indices
                # This might be memory heavy for small p, but N=10^6 is small.
                idx = np.arange(pk, N + 1, pk)
                
                # Filter out multiples of next_pk
                # idx % next_pk != 0
                # For p=2, pk=4, next_pk=8.
                # idx: 4, 8, 12, 16...
                # idx % 8: 4, 0, 4, 0...
                # mask: 1, 0, 1, 0...
                mask = (idx % next_pk) != 0
                P[idx[mask]] *= pk
            
            pk = next_pk
            k += 1
            
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
    
    # Find max P[n] * P[n+1] for n in [10000, N-1]
    # Vectorized max
    # Range is 10000 to N-1
    # P[n] is P[10000 : N]
    # P[n+1] is P[10001 : N+1]
    prods = P[10000:N] * P[10001:N+1]
    
    if len(prods) == 0:
        return 10000
        
    max_idx = np.argmax(prods)
    best_n = 10000 + max_idx
    
    return int(best_n)
