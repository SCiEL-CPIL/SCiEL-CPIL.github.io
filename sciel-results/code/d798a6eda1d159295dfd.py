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
            
            # Iterate through potential "last" primes i > j
            # Since primes[i] is increasing, gap_needed = primes[i] - primes[j] is increasing.
            # We can maintain a pointer into the sorted items to find the best k efficiently.
            ptr = 0
            best_len = 0
            best_k = -1
            num_items = len(items)
            
            for i in range(j + 1, N):
                gap_needed = primes[i] - primes[j]
                
                # Advance pointer while the gap of the item is <= gap_needed
                while ptr < num_items and items[ptr][0] <= gap_needed:
                    if items[ptr][1] > best_len:
                        best_len = items[ptr][1]
                        best_k = items[ptr][2]
                    ptr += 1
                
                if best_k != -1:
                    # We can extend a sequence ending at j
                    dp[i][j] = best_len + 1
                    prev[i][j] = best_k
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
