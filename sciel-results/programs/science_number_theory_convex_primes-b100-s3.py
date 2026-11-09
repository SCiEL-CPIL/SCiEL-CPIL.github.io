import time
import numpy as np

def construct():
    """
    Finds the longest subsequence of primes <= 10000 with non-decreasing gaps.
    Uses Dynamic Programming with NumPy for efficiency.
    """
    deadline = time.time() + 85.0
    LIMIT = 10000
    
    # 1. Generate primes using Sieve of Eratosthenes
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array([i for i, is_p in enumerate(sieve) if is_p], dtype=np.int32)
    N = len(primes)
    G = LIMIT # Maximum possible gap is LIMIT - 2
    
    # 2. Initialize DP arrays
    # dp[i][g] stores the maximum length of a valid sequence ending at primes[i] 
    # with the last gap exactly equal to g.
    # A sequence of length 1 has no gap, so we use gap=0 as a sentinel for length 1.
    dp = np.zeros((N, G + 1), dtype=np.int32)
    dp[:, 0] = 1  # Base case: any single prime is a valid sequence of length 1
    
    # pref[i][g] stores max(dp[i][0], ..., dp[i][g])
    # This allows O(1) lookup for "best predecessor with gap <= g"
    pref = np.zeros((N, G + 1), dtype=np.int32)
    
    # pred[i][g] stores the index j of the previous prime in the optimal sequence
    # ending at i with gap g. -1 indicates no predecessor (base case).
    pred = np.full((N, G + 1), -1, dtype=np.int32)
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    # 3. Fill DP table
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        
        # Transition: For each previous prime j, calculate the gap and update dp[i]
        for j in range(i):
            if time.time() > deadline:
                break
            gap = pi - primes[j]
            # The best sequence ending at j with gap <= 'gap' has length pref[j][gap]
            # Extending it gives length pref[j][gap] + 1
            val = pref[j][gap] + 1
            if val > dp[i][gap]:
                dp[i][gap] = val
                pred[i][gap] = j
                
        # Compute prefix maximums for row i
        pref[i] = np.maximum.accumulate(dp[i])
        
        # Update global best
        # np.argmax returns the index of the first occurrence of the max value
        max_len_i = np.max(dp[i])
        if max_len_i > best_len:
            best_len = max_len_i
            best_g = np.argmax(dp[i])
            best_i = i
            
    # 4. Reconstruct the sequence
    seq = []
    if best_len > 0:
        i = best_i
        g = best_g
        while i >= 0:
            seq.append(int(primes[i]))
            if i == 0:
                break
            j = pred[i][g]
            if j == -1:
                break
            # Find the specific gap h used in the predecessor state j
            # such that dp[j][h] == pref[j][g] and h <= g
            target_val = pref[j][g]
            h = g
            while h >= 0:
                if dp[j][h] == target_val:
                    break
                h -= 1
            i = j
            g = h
        seq.reverse()
        
    return seq
