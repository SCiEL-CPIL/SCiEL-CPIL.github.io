import math

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
    primes = _primes(N)
    # r[p] is the residue to exclude for prime p
    r = {p: 0 for p in primes}
    
    # cnt[x] is the number of primes p such that x % p == r[p]
    cnt = [0] * (N + 1)
    
    # Initialize cnt
    for p in primes:
        res = r[p]
        for x in range(res, N + 1, p):
            cnt[x] += 1
            
    # Initial size
    current_size = sum(1 for c in cnt if c == 0)
    
    # Coordinate Descent
    # We iterate until no improvement
    # To avoid getting stuck in local minima, we could do a few restarts or randomization,
    # but deterministic CD is a good baseline.
    
    # Optimization: Precompute lists of indices for each residue mod p?
    # No, N is small enough.
    
    improved = True
    while improved:
        improved = False
        for p in primes:
            old_r = r[p]
            best_r = old_r
            best_delta = 0
            
            # Calculate gain from removing old_r
            # x in old_r mod p: cnt[x] decreases by 1.
            # If cnt[x] was 1, it becomes 0 -> size increases by 1.
            gain = 0
            for x in range(old_r, N + 1, p):
                if cnt[x] == 1:
                    gain += 1
            
            # Try all other residues
            for new_r in range(p):
                if new_r == old_r:
                    continue
                # Calculate loss from adding new_r
                # x in new_r mod p: cnt[x] increases by 1.
                # If cnt[x] was 0, it becomes 1 -> size decreases by 1.
                loss = 0
                for x in range(new_r, N + 1, p):
                    if cnt[x] == 0:
                        loss += 1
                
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_r = new_r
            
            if best_r != old_r:
                # Apply change
                # Decrement cnt for old_r
                for x in range(old_r, N + 1, p):
                    cnt[x] -= 1
                # Increment cnt for new_r
                for x in range(best_r, N + 1, p):
                    cnt[x] += 1
                current_size += best_delta
                r[p] = best_r
                improved = True
    
    # Construct result
    result = [x for x in range(N + 1) if cnt[x] == 0]
    return result
