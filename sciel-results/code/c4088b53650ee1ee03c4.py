"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def compute_phi_segment(start, length, primes):
        """Compute phi for range [start, start+length-1] using given primes."""
        phi = np.arange(start, start + length, dtype=np.int64)
        max_val = start + length - 1
        for p in primes:
            if p * p > max_val:
                break
            first = ((start + p - 1) // p) * p
            indices = np.arange(first - start, length, p)
            mask = phi[indices] % p == 0
            if np.any(mask):
                idx_masked = indices[mask]
                phi[idx_masked] -= phi[idx_masked] // p
        # Vectorized final step: for remaining prime factors
        mask = phi > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // phi[mask]
        return phi
    
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    primes = sieve_primes(2000000)
    
    best_k = 1
    best_s = 1
    
    range_sizes = [200000, 500000, 1000000, 2000000, 5000000]
    start_positions = []
    # Generate start positions more systematically across the range
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for exp in range(6, 12):
            start_positions.append(base * 10**exp)
    # Add some intermediate positions
    for base in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]:
        for exp in range(6, 12):
            start_positions.append(int(base * 10**exp))
    start_positions = sorted(set(s for s in start_positions if s >= 1 and s <= 10**12 - 1))
    
    for size in range_sizes:
        if time.time() > deadline - 30:
            break
        for start in start_positions:
            if start + size > 10**12:
                continue
            if time.time() > deadline - 15:
                break
            
            phi_vals = compute_phi_segment(start, size, primes)
            
            # Sliding window with early termination
            seen = set()
            left = 0
            for right in range(size):
                val = phi_vals[right]
                while val in seen:
                    seen.discard(phi_vals[left])
                    left += 1
                seen.add(val)
                current_k = right - left + 1
                if current_k > best_k:
                    best_k = current_k
                    best_s = start + left
                # Early exit if we can't beat best_k
                if size - right <= best_k:
                    break
    
    # Verify the result
    if best_k > 1:
        verify_start = best_s
        verify_k = best_k
        seen = set()
        valid = True
        for i in range(verify_k):
            x = verify_start + i
            y = x
            result = x
            for p in primes:
                if p * p > y:
                    break
                if y % p == 0:
                    while y % p == 0:
                        y //= p
                    result -= result // p
            if y > 1:
                result -= result // y
            if result in seen:
                valid = False
                break
            seen.add(result)
        
        if not valid:
            return {'start': 1, 'length': 1}
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
