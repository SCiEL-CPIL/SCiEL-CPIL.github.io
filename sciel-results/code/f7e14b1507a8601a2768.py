import math
import time

def construct():
    start_time = time.time()
    LIMIT = 120.0 # seconds
    MAX_S = 10**12
    
    # 1. Generate primes up to 10^6
    # Sieve of Eratosthenes with bytearray for speed
    limit_p = 10**6
    is_prime = bytearray([1]) * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    # 2. Efficient phi calculation
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # 3. Search strategy
    # We want to maximize k.
    # Check blocks near 10^12.
    # Block size 10000.
    # If we find a run of 200, we can stop.
    
    best_k = 1
    best_s = 1
    
    # Check the last possible block first
    # We need s + k - 1 <= 10^12
    # Let's scan downwards from 10^12.
    
    block_size = 10000
    # Start from the largest possible start that allows a full block?
    # Or just slide a window?
    # Computing phi for 10000 numbers takes ~1s.
    # We have ~100s budget. Can do ~100 blocks.
    # 100 * 10000 = 10^6 range.
    
    # It is better to check the range [10^12 - 10^6, 10^12]
    # because collisions are rarest there.
    
    current_end = MAX_S
    while current_end > 1:
        # Check if we have time
        if time.time() - start_time > LIMIT - 10: # 10s margin
            break
            
        block_start = max(1, current_end - block_size + 1)
        
        # Compute phis for this block
        # We only need up to block_size elements
        # But we want to find a run of length k.
        # We can compute phis for the block.
        
        phis = []
        for i in range(block_start, current_end + 1):
            phis.append(phi(i))
            
        # Find longest run of distinct values in phis
        # Since we want to maximize k, and k<=200,
        # we can just scan.
        
        # To be efficient, we can look for runs ending at each position.
        # But simple nested loop is fine for B=10000, K=200.
        # Actually, we can just track a set as we go.
        
        seen = set()
        curr_k = 0
        max_k_in_block = 0
        start_pos = block_start
        
        # We need to find the start position of the best run.
        # A simple way: for each start in the block, extend as far as possible.
        # But that's O(B*K).
        # Better: iterate through the block, maintaining a sliding window of distinct values?
        # No, the condition is "consecutive integers".
        # So the window is fixed by the indices.
        
        # Let's just find the longest run.
        # We can use a two-pointer approach if we want O(B), but K is small.
        # Let's just do:
        # for i in range(len(phis)):
        #   if phis[i] in seen: reset? No, distinctness is required for the WHOLE run.
        #   So if phis[i] collides with any previous in the run, the run breaks.
        
        # We can maintain a set of values in the current run.
        # If collision, we need to shrink the window from the left.
        # This is the "longest substring without repeating characters" problem.
        
        l = 0
        r = 0
        seen_vals = {} # value -> index
        max_len = 0
        max_start_idx = 0
        
        while r < len(phis):
            if phis[r] in seen_vals and seen_vals[phis[r]] >= l:
                # Collision
                l = seen_vals[phis[r]] + 1
            else:
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
            
            seen_vals[phis[r]] = r
            r += 1
            
        # The best run in this block starts at block_start + max_start_idx
        # and has length max_len.
        
        # But we are limited by k <= 200.
        # If max_len > 200, we can just take 200.
        # The problem says 1 <= k <= 200.
        # So if we find a run of 300, we can return length 200.
        # We should return the actual length found, capped at 200?
        # "Maximize k". If a run of 300 exists, k=200 is achievable.
        # The evaluator measures the raw value k.
        # So if we find a run of 300, we should return k=200 (since max is 200).
        
        if max_len > best_k:
            if max_len >= 200:
                best_k = 200
                best_s = block_start + max_start_idx
                # We can return immediately as 200 is the max possible
                return {'start': best_s, 'length': best_k}
            else:
                best_k = max_len
                best_s = block_start + max_start_idx
        
        # Move to previous block
        current_end = block_start - 1
        
    return {'start': best_s, 'length': best_k}
