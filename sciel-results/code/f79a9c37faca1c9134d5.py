import math
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def now():
        return time.time()
        
    # Segmented sieve for phi
    def phi_segmented(L, R):
        """
        Compute phi(n) for n in [L, R].
        Returns a list of phi values.
        """
        size = R - L + 1
        if size <= 0:
            return []
        
        # num[i] holds the remaining part of (L+i) after dividing by small primes
        # phi[i] holds the current value of phi(L+i)
        num = [L + i for i in range(size)]
        phi = [L + i for i in range(size)]
        
        # Sieve primes up to sqrt(R)
        limit = int(math.isqrt(R)) + 1
        # Simple sieve for small primes
        is_prime = bytearray(b'\x01') * (limit + 1)
        if limit >= 0:
            is_prime[0] = 0
        if limit >= 1:
            is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = 0
        primes = [i for i in range(2, limit+1) if is_prime[i]]
        
        for p in primes:
            # Find the first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            for j in range(start, R + 1, p):
                idx = j - L
                # Update phi
                phi[idx] -= phi[idx] // p
                # Remove all factors of p from num
                while num[idx] % p == 0:
                    num[idx] //= p
                    
        # Finalize: if num[i] > 1, it is a prime factor
        for i in range(size):
            if num[i] > 1:
                phi[i] -= phi[i] // num[i]
                
        return phi

    best_start = 1
    best_len = 1
    
    def update_best(s, k):
        nonlocal best_start, best_len
        if k > best_len:
            best_len = k
            best_start = s

    # Search strategy
    # We search in blocks.
    # Block size 10^5 is fast. 10^6 is also fast.
    # Let's try blocks of 10^6.
    
    # First, ensure we check small numbers thoroughly as they often contain the longest runs
    # or at least competitive ones.
    
    block_size = 10**6
    
    # Search [1, 10^7]
    L = 1
    while L <= 10**7 and now() < deadline:
        R = min(L + block_size - 1, 10**7)
        vals = phi_segmented(L, R)
        
        # Find longest distinct run in this block
        # We need to be careful with boundaries if we want to combine blocks, 
        # but for simplicity, we check within the block first.
        # If a long run crosses a boundary, we might miss it.
        # To handle this, we can keep a tail of the previous block.
        
        # For now, let's just check within the block.
        # If we find a run of length k, it's valid.
        # To catch boundary runs, we can overlap blocks.
        
        # Overlapping strategy:
        # Keep the last 'max_possible_run' values from the previous block.
        # But we don't know max_possible_run yet.
        # Let's assume max run is < 1000.
        # So we can just re-check the last 2000 numbers of the previous block 
        # at the start of the next block?
        # Or simpler: just search [1, 10^7] in one go if memory allows?
        # 10^7 integers is 80MB. That's fine.
        
        # Let's do one big block [1, 10^7] first.
        if L == 1:
             R_big = 10**7
             vals_big = phi_segmented(1, R_big)
             # Slide window
             seen = set()
             cur_start = 1
             # This O(N*K) is too slow if K is large, but K is small.
             # Actually, we can just do a single pass with a set and reset on collision?
             # No, we need the longest run.
             # Standard approach:
             # for i in range(N):
             #   if vals[i] in seen: reset? No, that finds longest run of DISTINCT values, 
             #   but we need consecutive integers with distinct phi.
             #   The "seen" set must be cleared when we reset the start.
             
             # Efficient sliding window for longest distinct subarray:
             # But here the values are phi(n), not n.
             # We want max k such that phi(s)..phi(s+k-1) are distinct.
             
             # We can use a hash map of last seen index?
             # No, simpler:
             # Since K is small (likely < 50), we can just try all starts?
             # N=10^7, K=50. 5*10^8 operations. Might be too slow in Python.
             
             # Better: Sliding window with a set.
             # When we add a new element, if it's in the set, we shrink the window from the left 
             # until the duplicate is removed.
             
             window_start = 0
             seen_set = set()
             max_k_in_block = 0
             max_s_in_block = 1
             
             for i in range(len(vals_big)):
                 v = vals_big[i]
                 # If v is already in the set, we need to remove elements from the left
                 # until v is no longer in the set.
                 while v in seen_set:
                     seen_set.remove(vals_big[window_start])
                     window_start += 1
                 
                 seen_set.add(v)
                 current_len = i - window_start + 1
                 if current_len > max_k_in_block:
                     max_k_in_block = current_len
                     max_s_in_block = window_start + 1 # 1-based index
                     
             update_best(max_s_in_block, max_k_in_block)
             
             # After processing the big block, we can try larger ranges if time permits.
             # But 10^7 is likely sufficient.
             # Let's try [10^7, 2*10^7] if time permits.
             L = 10**7 + 1
             R = min(L + block_size - 1, 2*10**7)
             # To catch boundary effects, we need the last 2000 values of the previous block.
             # Let's just re-search [10^7 - 2000, 2*10^7] in one go?
             # That's 10^7 + 2000. Fine.
             
             if now() < deadline:
                 vals_big2 = phi_segmented(10**7 - 2000, 2*10**7)
                 window_start = 0
                 seen_set = set()
                 for i in range(len(vals_big2)):
                     v = vals_big2[i]
                     while v in seen_set:
                         seen_set.remove(vals_big2[window_start])
                         window_start += 1
                     seen_set.add(v)
                     current_len = i - window_start + 1
                     if current_len > max_k_in_block:
                         # Update global best
                         # Start index is (10^7 - 2000) + window_start
                         s = (10**7 - 2000) + window_start
                         if current_len > best_len:
                             best_len = current_len
                             best_start = s

    # If we have time, try a random large range?
    # Probably not needed.
    
    return {'start': best_start, 'length': best_len}
