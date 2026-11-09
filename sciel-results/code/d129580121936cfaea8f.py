import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    def get_primes_up_to(n):
        """Sieve of Eratosthenes to get primes up to n."""
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = sieve[1] = 0
        limit = int(math.isqrt(n))
        for i in range(2, limit + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]
    
    # We need to search for runs of consecutive integers with distinct totients.
    # The current seed finds a run of length ~14 in the first 20000.
    # We want to maximize k, up to 200, with s+k-1 <= 10^12.
    
    # Strategy:
    # 1. Efficiently compute phi(n) for a range of n.
    # 2. Scan for the longest run of consecutive n with distinct phi(n).
    # 3. Since 10^12 is huge, we can't scan all of it. We need to be smart about where to look.
    #    However, the problem says "Long runs below 10^12 are finite evidence".
    #    Let's first try to find good runs in a manageable range, say up to 10^6 or 10^7.
    #    If we find a run of length k, that's our answer. We want to maximize k.
    
    # Actually, let's think about the structure. phi(n) is even for n > 2.
    # So for any n > 2, phi(n) is even. This means phi(n) and phi(m) can only be distinct
    # if they are different even numbers.
    # Wait, the constraint is that phi(s), phi(s+1), ..., phi(s+k-1) are pairwise distinct.
    # Since all are even (for s > 2), they must be distinct even numbers.
    
    # Let's implement a segmented sieve for phi.
    # We'll compute phi for a segment [L, R] and check for distinct values in consecutive runs.
    
    def compute_phi_segment(L, R, primes):
        """Compute phi(n) for n in [L, R] using segmented sieve."""
        n_vals = R - L + 1
        phi_vals = list(range(L, R + 1))
        is_prime = [True] * n_vals
        is_prime[0] = False if L == 0 else True
        
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start == p:
                start += p  # p itself is prime, phi(p) = p-1, handled differently
            for j in range(start, R + 1, p):
                idx = j - L
                phi_vals[idx] -= phi_vals[idx] // p
                is_prime[idx] = False
        
        # For numbers that are prime, phi(p) = p - 1
        # But wait, the segmented sieve above already handles primes correctly if we initialize phi_vals[i] = i
        # and for each prime p, we do phi_vals[j] -= phi_vals[j] // p for all multiples j of p.
        # For a prime p in [L, R], it will only be touched by p itself (if p >= L), 
        # but actually in the loop, we start from the first multiple >= L.
        # If p is in [L, R], then p is a multiple of p, so phi_vals[p-L] -= phi_vals[p-L] // p = p - p/p = p-1. Correct.
        # But we also need to handle the case where L=1.
        if L == 1:
            phi_vals[0] = 1
        
        return phi_vals
    
    # Let's try to search in segments. We'll look for the longest run.
    # We need primes up to sqrt(10^12) = 10^6 for factoring, but for segmented phi,
    # we need primes up to R. Wait, no. For segmented sieve of phi, we need primes up to sqrt(R) 
    # to factor each number? No, actually the standard segmented phi sieve needs primes up to R 
    # to mark all multiples. But that's expensive.
    
    # Alternative: Use the fact that phi(n) = n * prod(1 - 1/p) for p|n.
    # For a segmented approach, we can use primes up to sqrt(R) to partially factor,
    # but we need to be careful.
    
    # Actually, a simpler approach: since we're limited to 120s, let's just compute phi 
    # for a range up to, say, 5 * 10^6 using a standard sieve, and find the best run.
    # Then if we have time, extend the range.
    
    # Let's first get primes up to 10^6 (sqrt(10^12)) for potential factoring, 
    # but for the sieve approach, we'll just use a direct sieve up to some limit.
    
    # Let's try a direct sieve for phi up to N = 5_000_000 first.
    N = 5_000_000
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Now find the longest run of consecutive n with distinct phi(n).
    best_k = 1
    best_s = 1
    
    # We can use a sliding window or just scan.
    # For each starting position s, extend k as long as phi(s+k) is not in {phi(s), ..., phi(s+k-1)}.
    # This is O(N * k) in the worst case, which is too slow for N=5e6.
    
    # Better: for each s, we want the maximum k such that phi(s), ..., phi(s+k-1) are distinct.
    # We can use a two-pointer / sliding window approach with a frequency map.
    
    from collections import defaultdict
    freq = defaultdict(int)
    left = 0
    current_max_k = 0
    for right in range(1, N + 1):
        freq[phi[right]] += 1
        while freq[phi[right]] > 1:
            freq[phi[left]] -= 1
            left += 1
        k = right - left + 1
        if k > current_max_k:
            current_max_k = k
            best_s = left
            best_k = k
    
    # Check time
    if time.time() > deadline:
        return {'start': best_s, 'length': best_k}
    
    # Now, can we do better by looking at larger ranges?
    # The maximum possible k is 200. If we found k close to 200, we're done.
    # If not, we might need to search further.
    # But note: phi(n) grows roughly as n, so in a range of length k, the phi values 
    # are spread over a range of size ~k * (average phi/n) ~ k * 0.6.
    # For distinctness, we need k distinct values in a range of size ~0.6k, which is impossible 
    # if k is large? No, phi(n) can be as small as ~n/log(log n), so the values are not 
    # uniformly spread.
    
    # Actually, for n > 2, phi(n) is even. So we're looking for k distinct even numbers 
    # in the sequence phi(s), ..., phi(s+k-1).
    # The minimum possible phi(n) for n in [s, s+k-1] is at least ~s/log(log s), 
    # and the maximum is s+k-1.
    # So the range of phi values is [~s/log(log s), s+k-1], which has size ~s.
    # We need k distinct even numbers in this range, which is possible as long as 
    # k <= s (roughly). So for large s, we can have large k.
    
    # Let's try to search in a larger range if we have time.
    # We'll use a segmented approach.
    
    # Get primes up to sqrt(10^12) = 10^6
    primes = get_primes_up_to(1000000)
    
    # Search in segments of size SEG_SIZE
    SEG_SIZE = 100000
    # Start from N+1 up to some limit
    search_limit = min(10**12 - 200, 50_000_000)  # don't go too far
    
    L = N + 1
    while L <= search_limit and time.time() < deadline:
        R = min(L + SEG_SIZE - 1, search_limit)
        phi_seg = compute_phi_segment(L, R, primes)
        
        # Find best run in this segment, but also consider runs that span segments.
        # For simplicity, let's just find the best run within this segment.
        # We can keep a window that spans the end of the previous segment.
        
        # For now, just scan within the segment.
        freq_seg = defaultdict(int)
        left_seg = 0
        for right_seg in range(len(phi_seg)):
            val = phi_seg[right_seg]
            freq_seg[val] += 1
            while freq_seg[val] > 1:
                freq_seg[phi_seg[left_seg]] -= 1
                left_seg += 1
            k_seg = right_seg - left_seg + 1
            if k_seg > best_k:
                best_k = k_seg
                best_s = L + left_seg
        
        L = R + 1
    
    return {'start': best_s, 'length': best_k}
