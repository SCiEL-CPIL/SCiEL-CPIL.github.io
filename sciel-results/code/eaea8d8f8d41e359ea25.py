import math

def construct():
    # Sieve of Eratosthenes to generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]
    
    # dp[i][j] = max length of valid sequence ending at primes[i] with previous prime primes[j]
    # prev[i][j] = index k such that primes[k] is the previous prime in the optimal sequence ending at primes[j] 
    #              that allows extending to primes[i]
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # Iterate through potential "second to last" primes j
    for j in range(N):
        if j == 0:
            # No previous prime k < 0
            # Any i > 0 can form a sequence of length 2 with j=0
            for i in range(1, N):
                dp[i][j] = 2
                prev[i][j] = -1
        else:
            # Prepare data for predecessor k < j
            # We need (gap, dp[j][k], k) for all k < j
            items = []
            for k in range(j):
                gap = primes[j] - primes[k]
                length = dp[j][k]
                items.append((gap, length, k))
            
            # Sort by gap
            items.sort(key=lambda x: x[0])
            
            # Use a pointer to maintain the best predecessor as gap_needed increases
            # This avoids binary search and prefix max arrays, replacing O(N log N) with O(N) per j
            p_idx = 0
            current_best_len = 0
            current_best_k = -1
            
            # Iterate through potential "last" primes i > j
            for i in range(j + 1, N):
                gap_needed = primes[i] - primes[j]
                
                # Update best predecessor if new items are valid
                while p_idx < len(items) and items[p_idx][0] <= gap_needed:
                    gap, length, k = items[p_idx]
                    if length > current_best_len:
                        current_best_len = length
                        current_best_k = k
                    p_idx += 1
                
                if current_best_len > 0:
                    # We can extend a sequence ending at j
                    dp[i][j] = current_best_len + 1
                    prev[i][j] = current_best_k
                else:
                    # No valid predecessor k, so start new sequence of length 2
                    dp[i][j] = 2
                    prev[i][j] = -1

    # Find the maximum length and the ending indices
    max_len = 0
    end_i = -1
    end_j = -1
    
    for i in range(N):
        for j in range(i):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_i = i
                end_j = j

    # If no pair was found (N < 2), handle separately, but N>=2 is guaranteed here
    if end_i == -1:
        return [primes[0]]

    # Backtrack to reconstruct the sequence
    path = [primes[end_i], primes[end_j]]
    i = end_i
    j = end_j
    
    while prev[i][j] != -1:
        k = prev[i][j]
        path.append(primes[k])
        i, j = j, k
        
    path.reverse()
    return path
