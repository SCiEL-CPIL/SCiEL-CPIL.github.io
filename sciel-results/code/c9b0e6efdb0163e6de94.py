"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import bisect

    # 1. Generate all primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    if n < 2:
        return primes

    best_len = 0
    best_seq = []

    # 2. Iterate over all possible starting pairs
    # We assume the greedy extension (smallest valid next gap) is optimal for length.
    for i in range(n - 1):
        # Pruning: max possible length starting at i is n - i
        if n - i <= best_len:
            break
            
        p_i = primes[i]
        for j in range(i + 1, n):
            # Pruning: max possible length starting at i, j is n - i (if we took all)
            # Actually, we have 2 elements so far. Max additional is n - j - 1.
            # Total max len = 2 + (n - j - 1) = n - j + 1.
            if n - j + 1 <= best_len:
                break

            p_j = primes[j]
            curr_gap = p_j - p_i
            
            # Greedy extension
            seq = [p_i, p_j]
            last_idx = j
            curr_prime = p_j
            curr_g = curr_gap
            
            # Extend as long as possible
            while True:
                # Find next prime p_k such that p_k - curr_prime >= curr_g
                # p_k >= curr_prime + curr_g
                target = curr_prime + curr_g
                # Find first prime >= target in primes starting from last_idx + 1
                k = bisect.bisect_left(primes, target, last_idx + 1)
                
                if k >= n:
                    break
                
                next_prime = primes[k]
                next_gap = next_prime - curr_prime
                
                # The condition next_gap >= curr_g is guaranteed by bisect_left
                # since next_prime >= curr_prime + curr_g
                
                seq.append(next_prime)
                curr_prime = next_prime
                curr_g = next_gap
                last_idx = k
            
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq

    return best_seq

# EVOLVE-BLOCK-END
