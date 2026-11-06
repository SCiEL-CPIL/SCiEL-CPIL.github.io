import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # Precompute square-free status for b up to max_b
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    
    # Sieve for square-free
    # We only need to check primes up to sqrt(max_b)
    for p in range(2, int(max_b ** 0.5) + 2):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s limit
    
    # We want to maximize the number of representations.
    # The number of representations of N as a sum of two powerful numbers
    # is roughly proportional to the density of powerful numbers up to N/2.
    # Generating more powerful numbers increases the chance of finding high-count N.
    
    # Let's try a limit that fits in memory and time for FFT.
    # Array size for FFT needs to be at least 2 * limit_pw.
    # 2 * 5e7 = 1e8. rfft of size 1e8 is feasible.
    
    limit_pw = 50_000_000  # 5e7
    
    # Check time budget for generation
    if time.time() - start_time > 10:
        # If generation is slow, reduce limit
        limit_pw = 20_000_000
        
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    
    elapsed_gen = time.time() - start_time
    if elapsed_gen > deadline - start_time - 30:
        # Not enough time for FFT on large array, try smaller or fallback
        limit_pw = 10_000_000
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)

    # Create a binary array for powerful numbers
    # Index i is 1 if i is powerful
    size = 2 * limit_pw + 10
    # Ensure size is good for FFT (power of 2 is fast, but numpy handles others)
    # We only care about sums up to 2 * limit_pw
    pw_array = np.zeros(size, dtype=np.float64)
    for p in powerful:
        if p < size:
            pw_array[p] = 1.0
    
    # Compute autocorrelation using FFT
    # This gives the number of pairs (a, b) such that a + b = k
    # Note: This counts ordered pairs (a,b) and (b,a) as distinct if a!=b,
    # and (a,a) once.
    # The task asks for distinct pairs a <= b.
    # If we use convolution of the indicator function with itself,
    # coeff[k] = sum_{i+j=k} A[i]A[j].
    # For a != b, both (a,b) and (b,a) contribute 1, so total 2.
    # For a == b, (a,a) contributes 1.
    # So if coeff[k] is even, number of a<=b pairs is coeff[k]/2.
    # If coeff[k] is odd, it means there is one (a,a) pair, so (coeff[k]-1)/2 + 1 = (coeff[k]+1)/2.
    
    fft_size = 1 << (size - 1).bit_length()  # Next power of 2
    if fft_size < size:
        fft_size = fft_size * 2
        
    # Perform FFT
    try:
        f = np.fft.rfft(pw_array, n=fft_size)
        # Power spectrum
        power = f * np.conj(f)
        # Inverse FFT
        inv = np.fft.irfft(power, n=fft_size)
        
        # The coefficients are in inv
        # We are interested in indices up to 2 * limit_pw
        max_sum = 2 * limit_pw
        
        # Find the index with the maximum count
        # Note: inv might have small negative values due to floating point errors
        inv_counts = inv[:max_sum + 1]
        
        # Find max
        max_idx = np.argmax(inv_counts)
        max_count_raw = int(round(inv_counts[max_idx]))
        
        # Calculate number of distinct pairs a <= b
        # If max_count_raw is even, pairs = max_count_raw // 2
        # If max_count_raw is odd, pairs = (max_count_raw + 1) // 2
        if max_count_raw % 2 == 0:
            best_count = max_count_raw // 2
        else:
            best_count = (max_count_raw + 1) // 2
            
        best_n = max_idx
        
        # Verify and collect pairs
        # We need to ensure N <= 10^18. Since limit_pw is 5e7, N <= 1e8, so this is fine.
        if best_n > 10**18:
             # Should not happen with our limits
             best_n = 2
             best_count = 1
             
        pairs = []
        # Iterate through powerful numbers to find actual pairs
        # This is O(M) where M is number of powerful numbers
        for a in powerful:
            if 2 * a > best_n:
                break
            b = best_n - a
            if b in powerful_set:
                pairs.append([a, b])
                
        # Double check count matches
        # If there's a mismatch due to FFT precision or logic, we might have issues.
        # But for binary inputs, FFT should be very accurate for integer counts.
        
        if pairs:
            return {'N': best_n, 'pairs': pairs}
            
    except Exception as e:
        # Fallback if FFT fails or memory error
        pass

    # Fallback: Brute force counting for smaller sets or if FFT failed
    # Use the powerful list we have
    n_pw = len(powerful)
    
    # If the set is small enough, do O(N^2) counting
    # 5e7 powerful numbers is too many for O(N^2).
    # But if we reduced limit_pw to 1e6, n_pw is ~1000, O(N^2) is 1e6 ops, fast.
    
    if n_pw < 50000:
        sum_count = {}
        for i in range(n_pw):
            a = powerful[i]
            for j in range(i, n_pw):
                b = powerful[j]
                s = a + b
                if s > 10**18:
                    break
                sum_count[s] = sum_count.get(s, 0) + 1
        
        if sum_count:
            best_n = max(sum_count, key=sum_count.get)
            best_count = sum_count[best_n]
            
            pairs = []
            for a in powerful:
                if 2 * a > best_n:
                    break
                b = best_n - a
                if b in powerful_set:
                    pairs.append([a, b])
            
            if pairs:
                return {'N': best_n, 'pairs': pairs}

    # Last resort: return a simple valid answer
    # 1 is powerful. 1+1=2.
    return {'N': 2, 'pairs': [[1, 1]]}
