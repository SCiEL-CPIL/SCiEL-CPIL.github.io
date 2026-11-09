import math
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill

    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi for all integers in [start, end] using vectorized factorization."""
        size = end - start + 1
        vals = np.arange(start, end + 1, dtype=np.int64)
        phi = vals.copy()
        rem = vals.copy()
        
        max_p = int(math.isqrt(end))
        
        for p in primes:
            if p > max_p:
                break
            mask = rem % p == 0
            if not mask.any():
                continue
            phi[mask] -= phi[mask] // p
            # Remove all factors of p from rem
            rem[mask] //= p
            while True:
                still_div = rem[mask] % p == 0
                if not still_div.any():
                    break
                rem[mask] = np.where(still_div, rem[mask] // p, rem[mask])
        
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
            
        return phi

    def find_longest_distinct_run(phi_vals):
        """Find the longest run of consecutive distinct values in phi_vals."""
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        
        best_k = 1
        best_s = 0
        seen = set()
        current_len = 0
        current_start = 0
        
        for i in range(n):
            val = int(phi_vals[i])
            if val in seen:
                seen = set()
                current_start = i
                current_len = 0
            seen.add(val)
            current_len += 1
            if current_len > best_k:
                best_k = current_len
                best_s = current_start
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Strategy: Use larger segments in high range where collisions are rare
    # Segment size of 2000 gives better chance of containing a run of 200
    seg_size = 2000
    max_start = 10**12 - seg_size + 1
    
    # Phase 1: Quick check of low range (cheap, might have small runs)
    try:
        phi_vals = compute_phi_segment(1, 1000)
        k, s = find_longest_distinct_run(phi_vals)
        if k > best_k:
            best_k = k
            best_s = 1 + s
    except Exception:
        pass

    # Phase 2: Random sampling in high range [10^10, 10^12]
    # High range has sparser totient values, so collisions are less likely
    random.seed(42)  # Reproducibility
    num_probes = 0
    
    while time.time() - start_time < deadline - 5:
        # Focus on high range where collisions are rare
        s_start = random.randint(10**10, max_start)
        
        try:
            phi_vals = compute_phi_segment(s_start, s_start + seg_size - 1)
            k, s = find_longest_distinct_run(phi_vals)
            actual_s = s_start + s
            if k > best_k:
                best_k = k
                best_s = actual_s
                if best_k >= 200:
                    break
            num_probes += 1
        except Exception:
            pass
    
    # Phase 3: If we haven't found a long run, try even higher range
    if best_k < 150 and time.time() - start_time < deadline - 5:
        while time.time() - start_time < deadline - 3:
            s_start = random.randint(10**11, max_start)
            try:
                phi_vals = compute_phi_segment(s_start, s_start + seg_size - 1)
                k, s = find_longest_distinct_run(phi_vals)
                actual_s = s_start + s
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    if best_k >= 200:
                        break
            except Exception:
                pass

    # Final safety checks
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
