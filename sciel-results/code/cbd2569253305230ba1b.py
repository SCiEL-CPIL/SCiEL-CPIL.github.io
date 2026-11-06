"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    # b^3 * a^2 <= limit
    # b up to limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # We will use FFT to find the N with the most representations.
    # To search for N up to N_max, we need an array size S >= 2 * N_max.
    # Let's try a large S.
    
    # Strategy: Try increasing FFT sizes.
    # Size 2^26 -> N_max 2^25
    # Size 2^27 -> N_max 2^26
    # Size 2^28 -> N_max 2^27 (might be too much memory: 2^28 * 8 bytes = 2GB)
    
    fft_sizes = [1 << 27, 1 << 26, 1 << 25]
    
    for size in fft_sizes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        # The max N we can reliably find is size // 2
        N_max = size // 2
        
        # Generate powerful numbers up to N_max
        powerful = _generate_powerful(N_max)
        powerful_set = set(powerful)
        
        if not powerful:
            continue
            
        # Create indicator array
        # Using float64 for precision. Counts are integers, so exact if < 2^53.
        A = np.zeros(size, dtype=np.float64)
        
        # Fill the array
        # powerful numbers are indices
        for p in powerful:
            if p < size:
                A[p] = 1.0
        
        # Perform FFT
        try:
            F = np.fft.fft(A)
            F2 = F * F
            C = np.fft.ifft(F2).real
        except MemoryError:
            continue
        except Exception:
            continue
            
        # C[n] is the number of ordered pairs (a,b) such that a+b=n
        # We want pairs a <= b.
        # Count = (C[n] + 1_{n even and n/2 powerful}) / 2
        
        # We are interested in N >= 2
        # Find max count
        # We restrict search to [2, N_max]
        
        if N_max >= 2:
            # Slice to avoid wrap-around effects if any (though size is 2*N_max, so sums up to 2*N_max fit)
            # Actually, if a,b <= N_max, a+b <= 2*N_max.
            # Our array size is 2*N_max.
            # Indices 0 to 2*N_max - 1 are valid.
            # We want N up to N_max.
            
            # Find the index with max value in range [2, N_max]
            # Note: C might have small numerical errors, but argmax should be robust.
            search_region = C[2:N_max+1]
            
            if len(search_region) == 0:
                continue
                
            max_idx_local = int(np.argmax(search_region))
            max_idx = max_idx_local + 2
            
            # Calculate actual number of pairs for this N
            N = max_idx
            
            # Verify and count exactly
            # C[N] is approximate count of ordered pairs.
            # Let's just count exactly to be safe and get pairs.
            
            count = 0
            pairs = []
            
            # Iterate through powerful numbers a <= N/2
            # We can break early if a > N/2
            half_N = N // 2
            
            for a in powerful:
                if a > half_N:
                    break
                b = N - a
                if b in powerful_set:
                    count += 1
                    pairs.append([a, b])
            
            if count > best_count:
                best_count = count
                best_result = {'N': N, 'pairs': pairs}
                
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # If we used the largest feasible size and found a good result, we can stop
        # Or we can try to refine?
        # The FFT gives a global max for that range.
        # A larger range is better.
        
    return best_result

# EVOLVE-BLOCK-END
