import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N)
    num_primes = len(primes)
    
    # counts[i][r]
    counts = [[0] * p for p in primes]
    
    S = set(range(N + 1))
    S_size = N + 1
    
    # Initialize counts
    for x in S:
        for i, p in enumerate(primes):
            counts[i][x % p] += 1
            
    violated = set()
    for i, p in enumerate(primes):
        if p <= S_size:
            # Check if all residues are present
            # min(counts[i]) > 0
            if min(counts[i]) > 0:
                violated.add(i)
                
    while violated:
        if time.time() > deadline:
            break
            
        # Prune violated set
        # Keep only i where primes[i] <= S_size
        violated = {i for i in violated if primes[i] <= S_size}
        
        if not violated:
            break
            
        # Find best i
        best_i = -1
        best_min = float('inf')
        best_r = -1
        
        for i in violated:
            p = primes[i]
            # Find min count
            # Using min with key
            min_val, min_r = min(enumerate(counts[i]), key=lambda x: x[1])
            
            if min_val < best_min:
                best_min = min_val
                best_i = i
                best_r = min_r
            elif min_val == best_min:
                # Tie break: larger prime
                if primes[i] > primes[best_i]:
                    best_i = i
                    best_r = min_r
                    
        if best_i == -1:
            break
            
        p_star = primes[best_i]
        r_star = best_r
        
        # Find elements to remove
        to_remove = []
        for x in S:
            if x % p_star == r_star:
                to_remove.append(x)
                
        for x in to_remove:
            S.remove(x)
            S_size -= 1
            
            # Update counts
            for i, p in enumerate(primes):
                r = x % p
                counts[i][r] -= 1
                if counts[i][r] == 0 and i in violated:
                    violated.discard(i)
                    
    return list(S)
