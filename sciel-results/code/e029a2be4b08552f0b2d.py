"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt
from collections import defaultdict

def generate_powerful(limit):
    """
    Generate all powerful numbers up to limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    pows = set([1])
    max_b = int(limit ** (1/3)) + 2
    
    # Sieve for square-free numbers is overkill for small max_b, 
    # but for larger limits, we might need efficiency.
    # However, max_b for limit=50M is ~368. Simple check is fine.
    
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        
        b_cubed = b ** 3
        if b_cubed > limit:
            break
            
        max_a = isqrt(limit // b_cubed)
        # Use list comprehension for speed in generation
        # a^2 * b^3
        # We can iterate a and add
        for a in range(1, max_a + 1):
            n = a * a * b_cubed
            if n <= limit:
                pows.add(n)
            else:
                break
                
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Use FFT convolution to count representations N = a + b.
    # 3. Find N with max count.
    # 4. Reconstruct pairs for that N.
    
    # Time budget allocation:
    # Generation: ~0.5s - 2s
    # FFT: ~1s - 5s (depends on size)
    # Reconstruction: ~0.1s
    # We can afford L up to ~50,000,000.
    # Number of powerful numbers up to 50M:
    # Density of powerful numbers is ~ C / sqrt(n).
    # Actually, count is ~ 2 * sqrt(n) / pi^2 * ... roughly.
    # Let's estimate: sqrt(50M) ~ 7071. Count is likely around 10k-20k.
    # K=20k -> M=40k. FFT size ~ 65536 or 131072. Very fast.
    
    # Let's push L higher to get more powerful numbers.
    # L = 100,000,000 (10^8)
    # sqrt(10^8) = 10,000.
    # Count of powerful numbers up to 10^8 is approx 2 * 10^4 / (pi^2) * ... 
    # Actually, the asymptotic is (2/sqrt(pi)) * n^{1/2} * ... no, it's sum_{k=1}^{n} 1/k^2 ...
    # Let's just run it.
    
    L = 10**8
    
    powerful_list = generate_powerful(L)
    powerful_set = set(powerful_list)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Create indicator vector for FFT
    # We need to compute C[N] = sum_{a+b=N} I(a) * I(b)
    # This is the convolution of the indicator vector with itself.
    # Since a and b are from the same set, and we want unordered pairs a<=b,
    # the convolution gives the number of ordered pairs (a,b) such that a+b=N.
    # For a != b, (a,b) and (b,a) are distinct in convolution, but same pair in our task.
    # For a == b, (a,a) is one pair.
    # Convolution count C[N] = 2 * (# pairs with a < b) + 1 * (# pairs with a = b)
    # Our task wants # pairs with a <= b.
    # Let X = # pairs a < b, Y = # pairs a = b.
    # C[N] = 2X + Y.
    # We want X + Y.
    # Note that Y is 1 if N is twice a powerful number, else 0.
    # So X = (C[N] - Y) / 2.
    # Score = X + Y = (C[N] - Y) / 2 + Y = (C[N] + Y) / 2.
    
    # We need to handle the case where N is even and N/2 is powerful.
    
    # FFT Size
    # Max N is 2*L.
    # Convolution length is 2*L + 1.
    # Next power of 2 for FFT.
    max_N = 2 * L
    fft_len = 1
    while fft_len <= max_N:
        fft_len <<= 1
        # Cap fft_len to avoid memory explosion if L is too big
        # 10^8 -> max_N 2*10^8. fft_len 2^28 ~ 268M.
        # Array of float64 is 8 bytes. 268M * 8 = 2GB. Too much for some machines.
        # Let's check available memory or reduce L if fft_len is huge.
        # 2^27 = 134M. 134M * 8 = 1GB. Still risky.
        # 2^26 = 67M. 67M * 8 = 536MB. Safe.
        # If fft_len > 2^26, we might need to reduce L.
        # For L=10^8, max_N=2*10^8. 2^27 is 134M < 200M. So fft_len will be 2^28?
        # 2^27 = 134,217,728.
        # 2*10^8 = 200,000,000.
        # So fft_len must be 2^28 = 268,435,456.
        # 268M complex128 is 16 bytes/element -> 4GB. TOO MUCH.
        
        # We need to reduce L or use a different approach.
        # Let's reduce L to 5*10^7.
        # max_N = 10^8.
        # fft_len = 2^27 = 134M.
        # 134M * 16 bytes = 2GB. Still high.
        
        # Let's try L = 2*10^7.
        # max_N = 4*10^7.
        # fft_len = 2^26 = 67M.
        # 67M * 16 bytes = 1GB. Might be okay, but tight.
        
        # Let's try L = 10^7.
        # max_N = 2*10^7.
        # fft_len = 2^25 = 33M.
        # 33M * 16 bytes = 512MB. Safe.
        
        # If we are at 2^26 or higher, we should break and reduce L.
        if fft_len > (1 << 25): # 33M
             # We are in the loop, we can't easily change L here without restarting.
             # Let's just pick a safe L initially.
             pass
             
    # To be safe and fast, let's stick to L = 10^7 or slightly higher.
    # The previous program used 10^7 and got 4.0.
    # With FFT, we can go higher.
    # Let's try L = 2*10^7.
    # max_N = 4*10^7.
    # fft_len = 2^26 = 67,108,864.
    # Memory: 67M * 16 bytes (complex128) = 1.07 GB.
    # Many evaluators have 4GB+ RAM. This should be fine.
    
    # Let's reset L to 2*10^7 for this run.
    # If it fails or times out, we'd need to adapt, but we can't adapt mid-run easily.
    # Let's be conservative: L = 15 * 10**6.
    # max_N = 30 * 10**6.
    # fft_len = 2^25 = 33,554,432.
    # Memory: 33M * 16 = 536 MB. Very safe.
    
    # Actually, let's just use L = 10**7 as in the seed, but with FFT it's much faster,
    # so we can afford L = 5 * 10**7?
    # max_N = 10**8.
    # fft_len = 2^27 = 134M.
    # Memory: 134M * 16 = 2.1 GB.
    # If the evaluator has 4GB, it works. If 2GB, it crashes.
    # Standard cloud instances often have 8-16GB.
    # Let's try L = 5 * 10**7.
    
    # Re-evaluating L based on typical constraints.
    # Let's try L = 30 * 10**6.
    # max_N = 60 * 10**6.
    # fft_len = 2^26 = 67M.
    # Memory ~ 1GB.
    
    # Let's stick to L = 20 * 10**6 to be safe on memory, but fast on time.
    L = 20 * 10**6
    
    # Regenerate powerful list for this L
    # (We already generated for 10^8 above, which is wasteful if we change L.
    # Let's structure the code to pick L first.)
    
    # RESTART LOGIC PROPERLY
    
    return construct_impl(L, start_time, deadline)

def construct_impl(L, start_time, deadline):
    powerful_list = generate_powerful(L)
    powerful_set = set(powerful_list)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    max_N = 2 * L
    fft_len = 1
    while fft_len <= max_N:
        fft_len <<= 1
        
    # If fft_len is too large, we might run out of memory.
    # 2^26 is ~67M. 2^27 is ~134M.
    # Let's assume 1GB-2GB is available.
    
    # Create vector
    # v[i] = 1 if i is in powerful_list else 0
    v = np.zeros(fft_len, dtype=np.complex128)
    # Map powerful numbers to indices
    # powerful_list contains values <= L
    # We can just iterate and set
    for p in powerful_list:
        v[p] = 1.0 + 0.0j
        
    # FFT
    t_start_fft = time.time()
    V = np.fft.fft(v)
    V_sq = V * V
    conv = np.fft.ifft(V_sq).real
    
    # Round to nearest integer
    conv = np.round(conv).astype(np.int64)
    
    # conv[N] is the number of ordered pairs (a,b) from powerful_list such that a+b=N.
    # We want unordered pairs a <= b.
    # Count = (conv[N] + is_self_pair(N)) / 2
    # where is_self_pair(N) is 1 if N is even and N/2 is in powerful_set, else 0.
    
    # We are interested in N such that both a and b are in powerful_list.
    # Since powerful_list only contains numbers <= L, 
    # if a+b=N, and a,b <= L, then N <= 2L.
    # Also a >= 1, b >= 1.
    
    # Find the best N
    # We need to check all N from 2 to 2L.
    # However, many N will have 0 representations.
    # We can just compute the score for all N and take the max.
    
    # Optimization:
    # Instead of checking all N, we can just find the max of (conv[N] + self_pair[N]) / 2.
    
    # Construct self_pair array
    # self_pair[N] = 1 if N % 2 == 0 and (N//2) in powerful_set
    
    # This check is O(2L) which is 40M for L=20M.
    # In Python, this loop might take 1-2 seconds.
    # We can vectorize it with NumPy.
    
    # Indices for even N
    even_indices = np.arange(2, max_N + 1, 2)
    # Halves
    halves = even_indices // 2
    
    # Check if halves are in powerful_set
    # powerful_set is a Python set. NumPy doesn't handle membership testing well directly.
    # We can create a boolean mask for powerful numbers up to L.
    # mask[p] = 1 if p in powerful_set
    
    mask = np.zeros(L + 1, dtype=np.uint8)
    for p in powerful_list:
        mask[p] = 1
        
    # Now, for each even N, check mask[N//2]
    # halves are indices into mask
    # Note: halves can be up to L.
    # mask is size L+1.
    
    self_pairs = mask[hales] if hasattr(mask, 'halves') else mask[hales] # Typo check
    # Actually:
    self_pairs = mask[hales]
    
    # Calculate scores
    # scores[N] = (conv[N] + self_pairs[N/2]) / 2
    # We only care about even N for the self_pair part.
    # For odd N, self_pair is 0.
    
    # Let's compute the total score array
    scores = np.zeros(max_N + 1, dtype=np.int64)
    
    # For all N, base score is conv[N] // 2 ? No.
    # (conv[N] + Y) / 2.
    # If N is odd, Y=0. conv[N] must be even?
    # If N is odd, a+b=N. Since a,b integers, one is even, one odd? No.
    # If a+b is odd, a!=b. So conv[N] = 2 * (# pairs).
    # So conv[N] is always even for odd N.
    # Score = conv[N] // 2.
    
    # If N is even, Y=1 if N/2 in powerful, else 0.
    # If Y=0, conv[N] is even (a!=b pairs). Score = conv[N] // 2.
    # If Y=1, conv[N] is odd (2X + 1). Score = (conv[N] + 1) // 2.
    
    # So:
    # scores = conv // 2
    # For even N where N/2 is powerful, scores[N] += 1
    
    scores = conv[:max_N+1] // 2
    
    # Add 1 for even N where N/2 is powerful
    # even_indices are the N's
    # self_pairs is the boolean array for those N's
    # We need to add 1 to scores at those indices
    
    # Create a boolean mask for scores
    # Or just iterate? 20M iterations in Python is slow.
    # Use NumPy fancy indexing.
    
    # indices_to_increment = even_indices[self_pairs == 1]
    # scores[indices_to_increment] += 1
    
    # But self_pairs is a NumPy array of 0/1.
    # We can do:
    scores[even_indices] += self_pairs
    
    # Find max
    # We want to maximize scores.
    # There might be ties. Any max is fine.
    best_N = int(np.argmax(scores))
    best_score = int(scores[best_N])
    
    # If best_score is 0, return fallback
    if best_score == 0:
         return {'N': 2, 'pairs': [[1, 1]]}
         
    # Reconstruct pairs for best_N
    pairs = []
    # Iterate a in powerful_list
    # a <= best_N / 2
    # b = best_N - a
    # if b in powerful_set and b >= a:
    #    pairs.append([a, b])
    
    # powerful_list is sorted.
    # We can break when a > best_N / 2.
    
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
            
    # Verify count
    # len(pairs) should be best_score.
    # If not, there's a bug in the convolution logic or reconstruction.
    # But we trust the convolution for the score.
    # The returned score is len(pairs).
    # If len(pairs) < best_score, we lose points.
    # This can happen if there are powerful numbers > L? No, we generated up to L.
    # If a+b=N, and a,b <= L, they are in the list.
    
    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
