import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We will search for k up to a certain limit N.
    # M = k * sigma(k) <= 10^18.
    # sigma(k) is roughly k * log(log(k)).
    # So k^2 * log(log(k)) <= 10^18.
    # k <= 10^9 / sqrt(log(log(k))).
    # For k=10^8, log(log(10^8)) is small, so k can be much larger.
    # However, memory is the constraint.
    # Let's try N = 10^7 first.
    
    N = 10**7
    
    # Sieve for sigma(k)
    # sigma is multiplicative.
    # We use a modified sieve.
    # sigma[i] = sum of divisors of i.
    
    sigma = [1] * (N + 1)
    
    # For each prime p, update sigma for multiples of p.
    # If i = p^a * m, gcd(p,m)=1, then sigma(i) = sigma(p^a) * sigma(m).
    # We can iterate p from 2 to N.
    
    # Optimization: Only iterate p if it is prime.
    # We can find primes using a simple sieve or check sigma[p]==1 initially?
    # Initially sigma[p]=1. If p is prime, sigma[p] should become 1+p.
    # But we need to know if p is prime to start the inner loop.
    
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
                
    # Now compute sigma using primes
    # sigma[i] starts at 1.
    # For each prime p, for each multiple m of p:
    #   Let m = p^a * r.
    #   sigma(m) = sigma(r) * (1 + p + ... + p^a).
    
    # We can do this by iterating p and updating sigma[m].
    # But we need to know the power of p in m.
    
    # Alternative:
    # sigma[1] = 1
    # for p in primes:
    #   for m in range(p, N+1, p):
    #     x = m
    #     p_power = 1
    #     term = 1
    #     while x % p == 0:
    #       x //= p
    #       p_power *= p
    #       term += p_power
    #     sigma[m] *= term
     
    # This is O(N log log N) roughly.
    # Let's check time for N=10^7.
    # Inner loop runs N/p times.
    # Sum of N/p for primes p is N log log N.
    # 10^7 * ~3 = 3*10^7 operations.
    # Each operation has a while loop.
    # This might be slow in Python.
    
    # Faster method:
    # Use the recurrence sigma(n) = sigma(n/p) * (1+p) + p^(a+1) ? No.
    # sigma(n) = sigma(n/p) + p^a * sigma(n/p^a) ?
    
    # Let's stick to the standard multiplicative update but optimize.
    # We can precompute the largest power of p dividing m?
    
    # Let's try a different approach:
    # Initialize sigma[i] = 1.
    # For p in primes:
    #   for m in range(p, N+1, p):
    #     # Find the highest power of p dividing m
    #     # This is expensive.
    
    # Let's use the property:
    # sigma is multiplicative.
    # We can compute spf (smallest prime factor) first.
    # Then sigma[n] = sigma[n // spf[n]^a] * (1 + spf[n] + ... + spf[n]^a)
    
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N+1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    sigma = [0] * (N + 1)
    sigma[1] = 1
    for i in range(2, N + 1):
        if time.time() > deadline - 20:
            break
        p = spf[i]
        # Find power of p in i
        x = i
        p_power = 1
        term = 1
        while x % p == 0:
            x //= p
            p_power *= p
            term += p_power
        # i = p^a * x, gcd(p,x)=1
        # sigma(i) = term * sigma(x)
        sigma[i] = term * sigma[x]
        
    # Now we have sigma[1..N]
    # Generate pairs (M, k)
    pairs = []
    limit_M = 10**18
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= limit_M:
            pairs.append((M, k))
        else:
            # Since M increases with k generally, we could break?
            # Not strictly monotonic, but mostly.
            # k * sigma[k] is roughly k^2 log log k.
            # So if M > 10^18, larger k will likely also be > 10^18.
            # But to be safe, we don't break, just skip.
            # Actually, we can break if k is large enough.
            # k^2 < 10^18 => k < 10^9.
            # Our N is 10^7, so all M might be < 10^18?
            # 10^7 * sigma(10^7) ~ 10^7 * 10^7 * 3 = 3*10^14 < 10^18.
            # So all pairs are valid.
            pass
            
    if time.time() > deadline - 10:
         # Return best found so far or seed
         return {'M': 333312, 'k': [336, 372, 434]}
         
    # Sort by M
    pairs.sort(key=lambda x: x[0])
    
    # Find M with max count
    best_M = 0
    best_ks = []
    current_M = -1
    current_ks = []
    max_count = 0
    
    for M, k in pairs:
        if M != current_M:
            if len(current_ks) > max_count:
                max_count = len(current_ks)
                best_M = current_M
                best_ks = current_ks
            current_M = M
            current_ks = [k]
        else:
            current_ks.append(k)
            
    if len(current_ks) > max_count:
        best_M = current_M
        best_ks = current_ks
        
    if best_M == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_ks}
