import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin
    
    best_size = 0
    best_set = []
    
    # Precompute primes up to N+1
    # We need primes up to max possible size.
    # Max size is <= N.
    
    # Helper to check and add
    def try_greedy(order):
        nonlocal best_size, best_set
        
        counts = {}
        zero_counts = {}
        # Initialize for primes we might encounter
        # We only need to track primes p <= current_size + 1
        # But x % p is needed for p up to current_size+1
        
        T = []
        
        for x in order:
            k = len(T)
            k_new = k + 1
            
            # Check validity
            valid = True
            # Iterate primes <= k_new
            # Optimization: only check primes that are <= k_new
            # We can precompute list of primes and break when p > k_new
            
            for p in PRIMES:
                if p > k_new:
                    break
                
                r = x % p
                if p not in counts:
                    # Initialize counts for this prime if not seen
                    # But we need zero_counts correctly.
                    # If p > k, counts are all 0.
                    # zero_count = p.
                    # If p == k_new, k = p-1.
                    # If we haven't seen p before, it means p > k (since we add primes as size grows).
                    # Actually, if p <= k, we must have seen it.
                    # So if p not in counts, then p > k.
                    # Since p <= k_new, p must be k_new (if k_new is prime) or p <= k (impossible if not in counts).
                    # Wait, if p <= k, we MUST have initialized it when size reached p-1?
                    # No, we initialize lazily.
                    # If p <= k, we should have counts.
                    # If not, it means we skipped it?
                    # We should initialize counts[p] when we first need it.
                    
                    # If p not in counts:
                    # It implies no elements in T are congruent to anything? No.
                    # It implies we haven't tracked this prime.
                    # This happens if p > previous max size.
                    # So p must be k_new (since p <= k_new).
                    # If p == k_new:
                    #   Current size k = p-1.
                    #   All counts are 0.
                    #   zero_counts[p] = p.
                    #   c = 0.
                    #   Check: c==0 and zero_counts[p]==1?
                    #   p == 1? No, p>=2.
                    #   So zero_counts[p] = p >= 2.
                    #   So it is SAFE.
                    #   We can add x.
                    #   Then we must update counts[p].
                    
                    # So if p not in counts:
                    #   It is safe.
                    #   We will update after.
                    continue 
                
                c = counts[p].get(r, 0)
                if c == 0:
                    if zero_counts[p] == 1:
                        valid = False
                        break
            
            if valid:
                T.append(x)
                # Update counts
                for p in PRIMES:
                    if p > k_new:
                        break
                    r = x % p
                    if p not in counts:
                        # Initialize
                        counts[p] = {}
                        zero_counts[p] = p
                        # Set count for r to 1
                        counts[p][r] = 1
                        zero_counts[p] -= 1
                    else:
                        c = counts[p].get(r, 0)
                        if c == 0:
                            zero_counts[p] -= 1
                        counts[p][r] = c + 1
                
                # Update best if larger
                if len(T) > best_size:
                    best_size = len(T)
                    best_set = T[:]
                    
                # Time check
                if time.time() > deadline:
                    return

    # Try multiple orderings
    # 1. Ascending
    try_greedy(range(N + 1))
    
    # 2. Descending
    if time.time() < deadline:
        try_greedy(range(N, -1, -1))
    
    # 3. Random restarts
    attempts = 0
    while time.time() < deadline:
        random.seed(attempts) # Reproducibility? No, just use random
        order = list(range(N + 1))
        random.shuffle(order)
        try_greedy(order)
        attempts += 1
        if attempts > 1000: break # Safety

    return best_set

