"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
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
    prime_set = set(primes)
    in_S = [True] * (N + 1)
    k = N + 1
    
    # Greedy Phase
    for p in primes:
        if p > k:
            break
        counts = [0] * p
        for v in range(N + 1):
            if in_S[v]:
                counts[v % p] += 1
        
        if min(counts) > 0:
            min_count = min(counts)
            target_class = counts.index(min_count)
            for v in range(N + 1):
                if in_S[v] and v % p == target_class:
                    in_S[v] = False
            k = sum(in_S)

    # Helper to compute critical primes
    def get_critical_primes():
        crit = {}
        for p in primes:
            if p > k:
                break
            counts = [0] * p
            for v in range(N + 1):
                if in_S[v]:
                    counts[v % p] += 1
            empty_classes = [i for i, c in enumerate(counts) if c == 0]
            if len(empty_classes) == 1:
                crit[p] = empty_classes[0]
            else:
                crit[p] = None
        return crit

    critical_primes = get_critical_primes()

    # Local Search Phase
    for x in range(N + 1):
        if in_S[x]:
            continue
        
        valid = True
        # Check existing primes
        for p, res in critical_primes.items():
            if res is not None and x % p == res:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check new prime if k+1 is prime
        new_k = k + 1
        if new_k in prime_set:
            res_counts = [0] * new_k
            for v in range(N + 1):
                if in_S[v]:
                    res_counts[v % new_k] += 1
            
            empty_new = [i for i, c in enumerate(res_counts) if c == 0]
            if len(empty_new) == 1:
                if x % new_k == empty_new[0]:
                    valid = False
            
            if valid:
                # Accept x
                in_S[x] = True
                k = new_k
                # Recompute critical primes
                critical_primes = get_critical_primes()
            else:
                # Reject x
                pass
        else:
            # Accept x
            in_S[x] = True
            k = new_k
            # Recompute critical primes
            critical_primes = get_critical_primes()

    return [v for v in range(N + 1) if in_S[v]]
# EVOLVE-BLOCK-END
