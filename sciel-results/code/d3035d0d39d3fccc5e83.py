import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    primes_np = np.array(primes, dtype=np.int64)

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1] using segmented sieve.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi values to the numbers themselves
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        # rem tracks the remaining part of the number to detect large prime factors
        rem = np.arange(start, start + length, dtype=np.int64)
        
        max_val = start + length - 1
        
        # Process small primes
        for p in primes:
            if p > max_val:
                break
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
            
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            
            # Remove all factors of p from rem
            for _ in range(40):  # Max exponent for p^k <= 10^12 is about 40 for p=2
                mask = rem[indices] % p == 0
                if not np.any(mask):
                    break
                rem[indices[mask]] //= p
        
        # Handle large prime factors
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = (phi_vals[mask] // rem[mask]) * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        """
        if length <= 0:
            return 0, start
        
        phis = compute_phi_segment(start, length)
        
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            if best_k >= max_k:
                break
        
        return min(best_k, max_k), best_s

    def collision_heuristic_score(start, k, primes_list):
        """
        Heuristic score for a candidate start: penalize if s+i is divisible by p
        where p-1 divides (i-j) for some pair, or simply if s+i is divisible by small primes
        that create phi collisions.
        
        Actually, a simpler and effective heuristic:
        phi(s+i) = phi(s+j) often happens if s+i and s+j share a large prime factor or have similar structure.
        A good proxy for "distinctness" is that the numbers s+i are not too "smooth" in a way that creates collisions.
        
        Let's use a different heuristic:
        For a fixed k, we want s such that phi(s), ..., phi(s+k-1) are distinct.
        Collisions phi(a)=phi(b) for a!=b are more likely if a and b have similar prime factorizations.
        
        A specific structural collision: phi(p) = p-1. If s+i is prime, phi(s+i) = s+i-1.
        If s+j-1 is also in the range and phi(s+j-1) = s+i-1, collision.
        
        Let's stick to the prompt's suggestion:
        "identifies a starting point s by ensuring s+i is not divisible by any prime p where p-1 divides a small difference d"
        
        This is complex to implement efficiently for large k.
        
        Alternative heuristic:
        Compute phi for a small window around candidate s, and check for duplicates.
        But that's what find_best_run_sliding does.
        
        Let's try a "local density" heuristic:
        For a candidate s, check if any s+i is divisible by a small prime p such that p-1 is small.
        If s+i is divisible by 2, phi(s+i) is even.
        If s+i is divisible by 3, phi(s+i) is divisible by 2.
        
        Actually, the most common collision is phi(n) = phi(n+1) or similar.
        
        Let's use a simple scoring:
        For candidate s, compute phi(s), phi(s+1), ..., phi(s+k-1) using a fast approximate method?
        No, exact factorization is required for verification.
        
        Let's use the heuristic:
        Prefer s such that s+i is NOT divisible by any prime p < 100.
        Why? Because if s+i is divisible by a small prime, it's more likely to have a "standard" phi value that might collide.
        Numbers with no small prime factors are more likely to be prime or have large prime factors, leading to more distinct phi values.
        
        Score = - (number of small prime factors of s+i for i in 0..k-1)
        """
        score = 0
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for i in range(k):
            n = start + i
            for p in small_primes:
                if n % p == 0:
                    score -= 1
                    # Count multiplicity?
                    while n % p == 0:
                        score -= 0.5
                        n //= p
        return score

    def find_best_candidate_heuristic(max_k=200):
        """
        Use heuristic to find promising start points.
        """
        best_score = -10000
        best_s = None
        
        # Search in a range
        search_start = 10**6
        search_end = 10**12 - max_k
        step = 10000  # Check every 10000 numbers
        
        # This is too slow if done naively. Let's sample.
        # Actually, the heuristic is cheap. Let's sample a few thousand candidates.
        candidates = []
        for _ in range(1000):
            s = random.randint(search_start, search_end)
            sc = collision_heuristic_score(s, max_k, primes)
            candidates.append((sc, s))
        
        candidates.sort(reverse=True)
        
        # Verify top candidates
        for sc, s in candidates[:20]:
            if time_left() < 5:
                break
            # Check a window around s
            window_start = max(1, s - 100)
            window_len = 400
            k, s_found = find_best_run_sliding(window_start, window_len, max_k)
            if k > best_k_global:
                best_k_global = k
                best_s_global = s_found
        
        return best_k_global, best_s_global

    best_k_global = 1
    best_s_global = 1
    
    # Baseline check
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 10000, 200)
        if k > best_k_global:
            best_k_global, best_s_global = k, s

    random.seed(42)
    
    # Adaptive search
    seg_len = 2000  # Start with larger segments to find longer runs
    
    # Phase 1: Random sampling with heuristic filtering
    while time_left() > 20:
        # Sample a candidate start
        start = random.randint(10**6, 10**12 - seg_len)
        
        # Quick heuristic check: is this region "dense" with small prime factors?
        # If yes, skip it.
        small_prime_count = 0
        for i in range(0, seg_len, 10):  # Sample every 10th number
            n = start + i
            for p in [2, 3, 5, 7, 11, 13]:
                if n % p == 0:
                    small_prime_count += 1
        
        # If too many small prime factors, this region might have many collisions.
        # But actually, we WANT to find regions with DISTINCT phi values.
        # Regions with many small prime factors might have MORE distinct phi values?
        # No, phi(n) for n with many small factors is "small" relative to n.
        # phi(n) for n prime is "large".
        # Collisions are more likely when phi values are close.
        
        # Let's just compute phi for a segment and check.
        k, s = find_best_run_sliding(start, seg_len, 200)
        if k > best_k_global:
            best_k_global, best_s_global = k, s
            
            # If we found a good run, try to extend it
            if best_k_global > 10 and time_left() > 10:
                # Extend left and right
                ext_start = max(1, best_s_global - 1000)
                ext_len = best_s_global - ext_start + best_k_global + 1000
                k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
                if k_ext > best_k_global:
                    best_k_global, best_s_global = k_ext, s_ext
                
                ext_start_r = best_s_global
                ext_end_r = min(10**12, best_s_global + best_k_global + 1000)
                ext_len_r = ext_end_r - ext_start_r + 1
                k_ext_r, s_ext_r = find_best_run_sliding(ext_start_r, ext_len_r, 200)
                if k_ext_r > best_k_global:
                    best_k_global, best_s_global = k_ext_r, s_ext_r

    # Phase 2: Heuristic-guided search
    if time_left() > 10:
        # Use the heuristic to find candidates
        for _ in range(50):  # Try 50 heuristic-guided searches
            if time_left() < 5:
                break
            
            # Generate candidates using heuristic
            candidates = []
            for _ in range(100):
                s = random.randint(10**6, 10**12 - 200)
                # Heuristic: prefer s where s+i is not divisible by small primes
                score = 0
                for i in range(200):
                    n = s + i
                    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                        if n % p == 0:
                            score -= 1
                candidates.append((score, s))
            
            candidates.sort(reverse=True)
            
            # Check top 5 candidates
            for sc, s in candidates[:5]:
                if time_left() < 5:
                    break
                k, s_found = find_best_run_sliding(s, 400, 200)
                if k > best_k_global:
                    best_k_global, best_s_global = k, s_found

    return {'start': best_s_global, 'length': best_k_global}
