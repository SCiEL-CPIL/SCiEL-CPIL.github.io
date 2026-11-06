import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s margin
    
    # Generate primes up to 10^6 (sqrt(10^12))
    # Actually, we only need primes up to sqrt(max_n) for the sieve to work correctly 
    # for the phi calculation in the segmented approach.
    # For a segment [L, R], we need primes p such that p^2 <= R.
    # If we process up to 10^12, we need primes up to 10^6.
    
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(np.sqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = False
        primes = np.where(is_prime)[0]
        return primes

    primes = sieve_primes(10**6)
    
    SEGMENT_SIZE = 10**6
    MAX_N = 10**12
    
    best_k = 1
    best_s = 1
    
    # We search sequentially
    # To be efficient, we can skip if the remaining time is small
    num_segments = int(MAX_N / SEGMENT_SIZE)
    
    for seg_idx in range(num_segments):
        if time.time() - start_time > DEADLINE:
            break
            
        start = seg_idx * SEGMENT_SIZE + 1
        end = min(start + SEGMENT_SIZE - 1, MAX_N)
        size = end - start + 1
        
        if size <= 0:
            break
            
        # Initialize arrays
        arr = np.arange(start, end + 1, dtype=np.int64)
        phi_arr = arr.copy()
        
        # Sieve phi
        for p in primes:
            if p * p > end:
                break
            mask = (arr % p == 0)
            if np.any(mask):
                phi_arr[mask] -= phi_arr[mask] // p
                
        # Handle the case where the remaining factor is prime (greater than sqrt(end))
        # This is handled implicitly if we only divide by primes p where p*p <= end.
        # Wait, the standard phi sieve:
        # phi(n) = n * prod (1 - 1/p)
        # If we divide by p for all p | n, we get the correct phi.
        # The condition p*p <= end ensures we catch all prime factors?
        # No. If n is prime, n > sqrt(end), then n is not divided by any p in the loop.
        # phi_arr remains n. But phi(n) = n-1.
        # So we must handle the case where the remaining value in phi_arr is > 1.
        # Actually, the standard segmented phi sieve works like this:
        # For each p, if arr[i] % p == 0, phi[i] -= phi[i] // p.
        # After processing all p <= sqrt(R), if phi[i] > 1, it means there is a prime factor > sqrt(R).
        # But wait, if n is prime, phi[n] should be n-1.
        # In the loop, if n is prime, no p divides it. phi_arr remains n.
        # We need to subtract 1? No, phi(p) = p-1.
        # The formula phi(n) = n * prod (1-1/p) works.
        # If n is prime, the product is empty? No, the product is over prime factors.
        # If n is prime, the factor is (1 - 1/n).
        # So phi(n) = n * (1 - 1/n) = n - 1.
        # My loop only processes p <= sqrt(end).
        # If n is prime, it is not processed. phi_arr[n] remains n.
        # We need to fix this.
        # Correction: After the loop, if phi_arr[i] > 1, it means there is a prime factor remaining?
        # No, if n = p*q, and p, q > sqrt(end), then n > end, impossible.
        # If n has a prime factor > sqrt(end), it can only have ONE such factor.
        # Let n = m * p, where p > sqrt(end). Then m < sqrt(end).
        # m is fully factored by the loop.
        # phi(n) = phi(m) * phi(p) = phi(m) * (p-1).
        # In the array, we have divided by factors of m.
        # So phi_arr[i] currently holds phi(m) * p?
        # Let's trace:
        # n = 10. p=2. phi[10] = 10 - 5 = 5.
        # p=3, 5. 10%5==0. phi[10] = 5 - 1 = 4.
        # Correct.
        # n = 7. p=2,3. 7%p != 0. phi[7] = 7.
        # But phi(7) = 6.
        # So we need to handle the case where the remaining number is prime.
        # How?
        # If phi_arr[i] > 1 and it was not modified by any p?
        # Actually, if n is prime, it is not modified.
        # If n = p * q with p < q, and p <= sqrt(n), q > sqrt(n).
        # p is processed. phi[n] becomes (n - n/p) = q * (p-1).
        # q is not processed (q > sqrt(end) is false if q is small, but if q > sqrt(end) then q > sqrt(n) which is impossible for q|n unless n is prime).
        # Wait, if n <= end, then sqrt(n) <= sqrt(end).
        # So any prime factor p of n satisfies p <= n <= end.
        # If p > sqrt(end), then p^2 > end >= n, so p cannot divide n (unless n=p, i.e., n is prime).
        # So the only case where a prime factor is not processed is if n itself is prime.
        # In that case, phi_arr[i] remains n.
        # We need to check if phi_arr[i] is prime? No, we just need to know if n was prime.
        # If n was prime, phi_arr[i] == n.
        # If n was composite, phi_arr[i] < n.
        # So: for all i where phi_arr[i] == arr[i], it means n was not divisible by any p <= sqrt(end).
        # This implies n is prime (or 1).
        # So we should set phi_arr[i] = phi_arr[i] - 1 for those indices.
        
        mask_prime = (phi_arr == arr)
        if np.any(mask_prime):
             phi_arr[mask_prime] -= 1
             
        # Now find longest run of distinct values
        # Convert to list for easier set operations? Or use numpy?
        # Since size is 10^6, converting to list is fast enough (~0.05s)
        vals = phi_arr.tolist()
        
        # Find max run
        current_max = 0
        # Optimization: if current_max >= best_k, we can stop checking windows that can't beat it?
        # But we need to find the *start* too.
        # Simple O(N*K) scan
        for i in range(size):
            # Early termination: if remaining elements <= best_k, break outer loop?
            # No, a run could start here and go far.
            # But if size - i <= best_k, we can break.
            if size - i <= best_k:
                break
                
            seen = set()
            length = 0
            for j in range(i, size):
                if vals[j] in seen:
                    break
                seen.add(vals[j])
                length += 1
                if length > best_k:
                    best_k = length
                    best_s = start + i
            
    return {'start': best_s, 'length': best_k}
