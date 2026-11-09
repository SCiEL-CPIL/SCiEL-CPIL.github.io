import bisect
import time

def construct():
    # Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N < 2:
        return primes
        
    # DP[i][j] = length of longest valid sequence ending at primes[i], primes[j]
    # Prev[i][j] = index k such that DP[k][i] was used to extend to DP[i][j]
    DP = [[0] * N for _ in range(N)]
    Prev = [[-1] * N for _ in range(N)]
    
    # SufMax[j][i] = max(DP[m][j] for m in [i, j-1])
    # SufArg[j][i] = argmax index
    # We will build these on the fly for each j
    
    max_len = 0
    best_i, best_j = -1, -1
    
    start_time = time.time()
    
    for j in range(N):
        # Build suffix max for column j (values DP[i][j] for i < j)
        if j > 0:
            suf_max = [0] * j
            suf_arg = [-1] * j
            # Initialize last element
            suf_max[j-1] = DP[j-1][j]
            suf_arg[j-1] = j-1
            for i in range(j-2, -1, -1):
                if DP[i][j] > suf_max[i+1]:
                    suf_max[i] = DP[i][j]
                    suf_arg[i] = i
                else:
                    suf_max[i] = suf_max[i+1]
                    suf_arg[i] = suf_arg[i+1]
        else:
            suf_max = []
            suf_arg = []
            
        # Update DP[j][k] for k > j
        for k in range(j+1, N):
            # Check deadline
            if time.time() - start_time > 50:
                # Return best found so far
                if max_len == 0:
                    return [primes[0], primes[1]] if N > 1 else primes
                # Reconstruct best
                seq = [primes[best_i], primes[best_j]]
                i, jj = best_i, best_j
                while Prev[i][jj] != -1:
                    pi = Prev[i][jj]
                    seq.append(primes[pi])
                    jj = i
                    i = pi
                seq.reverse()
                return seq
                
            T = 2 * primes[j] - primes[k]
            # Find smallest index idx in [0, j-1] such that primes[idx] >= T
            idx = bisect.bisect_left(primes, T, 0, j)
            
            if idx < j:
                # Valid range is [idx, j-1]
                # Max length from suf_max[idx]
                candidate_len = suf_max[idx] + 1
                candidate_prev = suf_arg[idx]
                
                if candidate_len > DP[j][k]:
                    DP[j][k] = candidate_len
                    Prev[j][k] = candidate_prev
            else:
                # No valid predecessor i, so length is 2
                if DP[j][k] < 2:
                    DP[j][k] = 2
                    Prev[j][k] = -1
            
            # Also, the base case of length 2 is always valid
            # But DP[j][k] is initialized to 0.
            # If idx >= j, we set it to 2.
            # If idx < j, candidate_len is at least 2+1=3?
            # Wait. DP[i][j] is at least 2?
            # Yes, any pair (i,j) forms a sequence of length 2.
            # So suf_max[idx] is at least 2.
            # So candidate_len is at least 3.
            # So if idx < j, we always get length >= 3.
            # If idx >= j, we get length 2.
            
            if DP[j][k] > max_len:
                max_len = DP[j][k]
                best_i, best_j = j, k

    # Reconstruct the best sequence
    if max_len == 0:
        return primes[:2] if N >= 2 else primes
        
    seq = [primes[best_i], primes[best_j]]
    i, jj = best_i, best_j
    while Prev[i][jj] != -1:
        pi = Prev[i][jj]
        seq.append(primes[pi])
        jj = i
        i = pi
    seq.reverse()
    
    return seq

