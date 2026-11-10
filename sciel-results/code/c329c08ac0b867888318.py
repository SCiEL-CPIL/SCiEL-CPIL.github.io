# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Generate primes <= 10000
    n = 10000
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, n + 1) if sieve[i]]
    num_primes = len(primes)
    
    # best[i]: dict mapping last_gap -> max_length
    # parent[i]: dict mapping last_gap -> (prev_prime_idx, prev_last_gap)
    best = [dict() for _ in range(num_primes)]
    parent = [dict() for _ in range(num_primes)]
    
    for i in range(num_primes):
        best[i][0] = 1
        parent[i][0] = (-1, 0)
    
    max_len = 1
    best_end = 0
    best_gap = 0
    
    for i in range(1, num_primes):
        if time.time() > deadline:
            break
        for j in range(i):
            gap = primes[i] - primes[j]
            # Iterate over best[j]
            for prev_gap, length in best[j].items():
                if prev_gap <= gap:
                    new_len = length + 1
                    cur = best[i].get(gap, 0)
                    if new_len > cur:
                        best[i][gap] = new_len
                        parent[i][gap] = (j, prev_gap)
                        if new_len > max_len:
                            max_len = new_len
                            best_end = i
                            best_gap = gap
    
    if time.time() > deadline:
        # Greedy fallback
        values = []
        for p in primes:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values
    
    # Backtrack
    seq = []
    i = best_end
    gap = best_gap
    while i != -1:
        seq.append(primes[i])
        if parent[i][gap] == (-1, 0):
            break
        i, gap = parent[i][gap]
    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[k] - seq[k-1] for k in range(1, len(seq))]
        for k in range(1, len(gaps)):
            if gaps[k] < gaps[k-1]:
                # Invalid, use greedy
                values = []
                for p in primes:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                return values
    
    return seq

# EVOLVE-BLOCK-END
