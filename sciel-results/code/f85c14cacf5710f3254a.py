import time
import math
from bisect import bisect_left

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # 55 seconds safety margin
    
    N = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    M = len(primes)
    
    if M < 2:
        return primes
    
    # dp[i][j] = max length of sequence ending with primes[j], primes[i]
    # We only need to store dp[i][j] for j < i
    # To save memory and time, we can store as a list of lists
    # dp[i] is a list of length i (for j=0..i-1)
    dp = [None] * M
    dp[0] = []
    dp[1] = [2] # dp[1][0] = 2
    
    global_max = 2
    
    # Precompute suffix max for each row j?
    # We need suf_max[j][k] = max(dp[j][k], ..., dp[j][j-1])
    # We can compute this on the fly for each j before using it.
    
    for j in range(0, M - 1):
        # Build suffix max array for row j
        # dp[j] has length j
        # suf_max[k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
        suf_max = [0] * (j + 1)
        if j > 0:
            suf_max[j-1] = dp[j][j-1]
            for k in range(j-2, -1, -1):
                suf_max[k] = max(dp[j][k], suf_max[k+1])
        
        # Initialize k_curr
        # T = 2*P[j] - P[i]
        # For i = j+1, T = 2*P[j] - P[j+1]
        T = 2 * primes[j] - primes[j+1]
        k_curr = bisect_left(primes, T)
        if k_curr > j:
            k_curr = j
        
        # Prepare dp[i] for i > j
        # We will fill dp[i][j]
        
        for i in range(j + 1, M):
            if time.time() - start_time > DEADLINE:
                # Return best found so far
                # We need to reconstruct the sequence?
                # The problem asks to return the sequence.
                # We only tracked the max length.
                # We need to store backpointers or re-run?
                # Storing backpointers adds memory but is safe.
                # Or we can just return a valid sequence of length global_max?
                # The task says "Return increasing prime integers...".
                # It implies we should return the actual sequence.
                # If we timeout, we should return the best sequence we have.
                # But we didn't store the sequence, just the length.
                # Let's assume we don't timeout. M=1229, M^2=1.5M, very fast.
                pass
            
            T = 2 * primes[j] - primes[i]
            # Update k_curr
            # T decreases as i increases
            while k_curr > 0 and primes[k_curr - 1] >= T:
                k_curr -= 1
            
            if k_curr < j:
                val = suf_max[k_curr] + 1
            else:
                val = 2
            
            # Store dp[i][j]
            if dp[i] is None:
                dp[i] = [0] * i
            dp[i][j] = val
            
            if val > global_max:
                global_max = val
    
    # Reconstruct the sequence
    # Find the pair (i, j) that gives global_max
    best_i, best_j = -1, -1
    for i in range(M):
        for j in range(i):
            if dp[i][j] == global_max:
                best_i = i
                best_j = j
                break
        if best_i != -1:
            break
    
    if best_i == -1:
        return primes[:global_max] # Fallback
    
    seq = [primes[best_i], primes[best_j]]
    # Trace back
    cur_i, cur_j = best_i, best_j
    while cur_j > 0:
        # We need to find k such that dp[cur_j][k] = dp[cur_i][cur_j] - 1
        # And primes[cur_j] - primes[k] <= primes[cur_i] - primes[cur_j]
        target_len = dp[cur_i][cur_j] - 1
        gap = primes[cur_i] - primes[cur_j]
        
        # Find k
        for k in range(cur_j):
            if dp[cur_j][k] == target_len and (primes[cur_j] - primes[k] <= gap):
                seq.append(primes[k])
                cur_i, cur_j = cur_j, k
                break
        else:
            break # Should not happen
    
    seq.reverse()
    return seq

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
    # Verify
    gaps = [res[i+1] - res[i] for i in range(len(res)-1)]
    for i in range(len(gaps)-1):
        assert gaps[i] <= gaps[i+1], f"Gap violation at {i}: {gaps[i]} > {gaps[i+1]}"
    assert all(res[i] < res[i+1] for i in range(len(res)-1))
    assert all(p <= 10000 for p in res)
    # Check primality
    def is_prime_check(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0: return False
        return True
    assert all(is_prime_check(p) for p in res)
