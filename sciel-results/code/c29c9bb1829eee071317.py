import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 60s

    # Generate primes up to 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(10000)
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]

    # dp[i][j] will store the length of the longest sequence ending with primes[i], primes[j]
    # We use a 2D list of lists. N is ~1229, so N*N is ~1.5M, which fits in memory.
    # Initialize with 0.
    dp = [[0] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    max_len = 1
    best_i = 0
    best_j = 0

    # We iterate i from 0 to N-1
    # For each i, we assume dp[h][i] is already computed for all h < i
    # We precompute suffix max info for column i
    # Then we compute dp[i][j] for all j > i
    
    for i in range(N):
        if time.time() > deadline:
            break
            
        # Precompute suffix max for column i (rows 0 to i-1)
        # We want, for any threshold T, the index h in [0, i-1] with primes[h] >= T
        # that maximizes dp[h][i]
        
        # If i == 0, there are no h < i.
        if i > 0:
            # Create arrays for suffix max value and index
            # suf_val[h] = max(dp[m][i] for m in h..i-1)
            # suf_idx[h] = argmax
            suf_val = [0] * i
            suf_idx = [-1] * i
            
            # Start from i-1 down to 0
            max_val = -1
            max_idx = -1
            for h in range(i - 1, -1, -1):
                if dp[h][i] > max_val:
                    max_val = dp[h][i]
                    max_idx = h
                suf_val[h] = max_val
                suf_idx[h] = max_idx
        else:
            suf_val = []
            suf_idx = []

        # Now compute dp[i][j] for j in i+1 .. N-1
        for j in range(i + 1, N):
            if time.time() > deadline:
                break
                
            # Condition: primes[i] - primes[h] <= primes[j] - primes[i]
            # primes[h] >= 2 * primes[i] - primes[j]
            T = 2 * primes[i] - primes[j]
            
            if i == 0:
                # No previous prime h exists
                dp[i][j] = 2
                parent[i][j] = -1
            else:
                # Find smallest h in [0, i-1] such that primes[h] >= T
                # Use bisect_left on primes[0:i]
                # We need to search in the list primes[0..i-1]
                # bisect_left returns the index in the full list if we search the slice?
                # No, bisect works on the list. We can use bisect_left(primes, T, 0, i)
                import bisect
                h_min = bisect.bisect_left(primes, T, 0, i)
                
                if h_min < i:
                    # There is a valid h
                    # The best h is suf_idx[h_min]
                    best_h = suf_idx[h_min]
                    dp[i][j] = dp[best_h][i] + 1
                    parent[i][j] = best_h
                else:
                    # No valid h, sequence starts at i
                    dp[i][j] = 2
                    parent[i][j] = -1
            
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j

    # Reconstruct the path
    # The path ends at (best_i, best_j)
    # We need to backtrack to the start
    seq = []
    if max_len >= 2:
        seq.append(primes[best_j])
        seq.append(primes[best_i])
        
        curr_i = best_i
        curr_j = best_j
        # Backtrack while parent exists
        while parent[curr_i][curr_j] != -1:
            prev_h = parent[curr_i][curr_j]
            seq.append(primes[prev_h])
            curr_j = curr_i
            curr_i = prev_h
            
        seq.reverse()
    else:
        # If max_len is 1 (unlikely for N>1), just return first prime
        seq = [primes[0]]

    return seq

if __name__ == "__main__":
    print(construct())
