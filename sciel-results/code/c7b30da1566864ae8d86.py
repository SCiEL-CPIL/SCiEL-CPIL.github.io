import time
import numpy as np

def construct():
    start_time = time.time()
    timeout = 140  # seconds, leave margin for 150s limit
    
    # Strategy:
    # 1. Generate powerful numbers up to a limit M.
    #    Powerful numbers are of the form u^2 * v^3.
    #    The count of powerful numbers up to M is approx 0.65 * M^(2/3).
    # 2. Use FFT/Convolution to find N with the maximum number of pairs (a,b) 
    #    such that a+b=N and a,b are powerful.
    #    This allows us to search a much larger range of N than brute force.
    
    # Determine M based on time budget.
    # Generation is O(M^(1/3) * M^(1/3)) roughly? No, u^2 v^3 <= M.
    # v up to M^(1/3). For each v, u up to sqrt(M/v^3).
    # Total powerful numbers K ~ M^(2/3).
    # FFT cost is O(N_fft * log(N_fft)).
    # We need N_fft >= 2*M.
    
    # Let's try M = 10^9.
    # K ~ 0.65 * (10^9)^(2/3) = 0.65 * 10^6 = 650,000.
    # N_fft ~ 2 * 10^9. This is too large for standard FFT in Python/NumPy within 150s?
    # Actually, 2 * 10^9 complex numbers is ~ 32 GB memory. Too much.
    
    # Let's scale down.
    # M = 10^8.
    # K ~ 0.65 * 10^(16/3) ~ 0.65 * 4.6 * 10^5 ~ 300,000.
    # N_fft ~ 2 * 10^8. Memory ~ 3.2 GB. Might be tight but possible if optimized?
    # NumPy FFT on 2^27 (~1.3e8) or 2^28 (~2.6e8) points.
    # 2^28 points is 268M. 268M * 16 bytes (complex128) = 4.3 GB.
    # This is likely to cause memory issues or be slow.
    
    # Let's try M = 2 * 10^7.
    # K ~ 0.65 * (2*10^7)^(2/3) ~ 0.65 * 74000 ~ 48,000.
    # N_fft ~ 4 * 10^7.
    # Next power of 2: 2^26 = 67,108,864.
    # Memory: 67M * 16 bytes = ~1 GB. This is safe.
    # FFT time: O(N log N) ~ 67M * 26 ~ 1.7B ops. NumPy FFT is fast (C implementation).
    # Should take < 10 seconds.
    
    M = 20_000_000
    
    # Generate powerful numbers up to M
    powerful_set = set()
    max_v = int(M ** (1/3)) + 1
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = int(np.sqrt(M // v_cubed))
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # Prepare array for FFT
    # We need to compute convolution of indicator function of powerful numbers.
    # Let A be an array of length 2*M where A[x] = 1 if x is powerful, else 0.
    # Then C = A * A (convolution).
    # C[n] = sum_{i+j=n} A[i]A[j] = number of pairs (i,j) such that i+j=n and both powerful.
    # Note: This counts ordered pairs (a,b) and (b,a) as distinct if a!=b.
    # The task asks for distinct pairs a <= b.
    # If a != b, (a,b) and (b,a) are the same unordered pair.
    # If a == b, (a,a) is one pair.
    # So, count_unordered = (C[n] + delta_{n even and n/2 powerful}) / 2.
    # Specifically:
    # C[n] = 2 * (# pairs a<b) + 1 * (# pairs a=b)
    # Let X = # pairs a<b, Y = # pairs a=b (Y is 1 if n/2 is powerful and n even, else 0).
    # C[n] = 2X + Y.
    # We want X + Y.
    # X = (C[n] - Y) / 2.
    # X + Y = (C[n] - Y) / 2 + Y = (C[n] + Y) / 2.
    
    # Determine FFT size
    # We need indices up to 2*M.
    # Size must be >= 2*M + 1.
    # Use next power of 2 for speed.
    size = 1
    while size < 2 * M + 1:
        size *= 2
        
    # Create array
    # Using float64 is usually sufficient for counts < 10^6?
    # Max count is K ~ 50,000. Squared is 2.5 * 10^9.
    # Float64 has 52 bits of precision, so integers up to 2^53 are exact.
    # So float64 is safe.
    
    arr = np.zeros(size, dtype=np.float64)
    for p in powerful_list:
        arr[p] = 1.0
        
    # Perform FFT
    fft_arr = np.fft.fft(arr)
    # Square in frequency domain
    fft_arr *= fft_arr
    # Inverse FFT
    conv = np.fft.ifft(fft_arr).real
    
    # conv[n] should be integer (count of ordered pairs).
    # Due to floating point errors, we round to nearest integer.
    # We are only interested in n <= 2*M.
    # Note: conv array is size `size`. We only care up to 2*M.
    
    # Find the best N
    best_N = 0
    best_score = 0
    
    # We need to calculate the score (number of unordered pairs) for each N.
    # score(N) = (conv[N] + Y(N)) / 2
    # Y(N) = 1 if N is even and N/2 is powerful, else 0.
    
    # To optimize, we can just find the max conv[N] first?
    # Not exactly, because of the Y term, but Y is at most 1, while conv can be large.
    # The difference between two N's conv values is likely > 1 if they are distinct peaks.
    # However, to be safe, we should compute the exact score for the top candidates.
    
    # Get top 100 candidates by conv[N]
    # We only consider N in [2, 2*M]
    # Note: conv[0] corresponds to N=0 (not allowed, N>=2).
    
    # Create a view of conv[2 : 2*M + 1]
    # Indices in conv array: 0 to size-1.
    # N ranges from 2 to 2*M.
    
    # For efficiency, find max in conv[2:2*M+1]
    # But we need the exact score.
    
    # Let's just iterate through the top K candidates from conv.
    # Or simply iterate through all N? 40M iterations in Python is too slow.
    # We need to use NumPy to find the max.
    
    # conv_val = conv[2:2*M+1]
    # max_idx = np.argmax(conv_val)
    # best_N = max_idx + 2
    
    # But we must account for the Y term.
    # Let's find the top 20 indices with highest conv values.
    
    # To avoid sorting 40M elements, we can use np.argpartition or just check a range.
    # Actually, the maximum is likely unique or close.
    
    # Let's extract the relevant slice
    # Note: conv is complex? No, .real taken.
    
    # We need to be careful with memory if we slice a large array.
    # conv is already in memory.
    
    # Find the index of the maximum value in conv[2:2*M+1]
    # We can use np.argmax on the slice.
    # This creates a view, so it's fast.
    
    # However, we want to be robust against floating point noise.
    # Let's find the top 10 candidates.
    
    # np.argpartition returns indices of the k largest elements.
    # k=10.
    
    # slice_conv = conv[2:2*M+1]
    # part = np.argpartition(slice_conv, -10)[-10:]
    # candidates = part + 2
    
    # Then evaluate score for these 10 candidates.
    
    # Let's do this.
    
    # Check time
    if time.time() - start_time > timeout - 10:
         # Fallback: just return the max conv index without precise scoring?
         # Or return a known good N?
         # Let's just proceed, FFT should be fast.
         pass

    # Slice view
    # Note: conv is length `size`. 2*M is within `size`.
    # We want N from 2 to 2*M.
    
    # Get top 20 candidates
    # If 2*M+1 is large, argpartition is O(N).
    
    # Let's just find the global max of conv in the valid range.
    # max_val = np.max(conv[2:2*M+1])
    # indices = np.where(conv[2:2*M+1] == max_val)[0]
    
    # This might be slow if there are many ties? Unlikely.
    
    # Let's use argpartition for top 10.
    # k=10
    # valid_slice = conv[2:2*M+1]
    # top_k_indices = np.argpartition(valid_slice, -10)[-10:]
    
    # But wait, conv values are floats. We should round them to integers for comparison?
    # The differences are integers.
    # Let's just use the float values.
    
    # One issue: np.argpartition on a slice of a large array might copy?
    # No, slice is a view. argpartition works on the view.
    
    # Let's try to be safe and just find the argmax.
    # If there are ties, we might miss a better one with Y=1 vs Y=0?
    # Difference is 0.5 in score.
    # If conv[N1] = 100, Y=0 -> Score 50.
    # If conv[N2] = 100, Y=1 -> Score 50.5.
    # If we pick N1, we get 50. If we pick N2, we get 50.5.
    # Since score is integer, 50.5 is actually 50 (floor?) or 51?
    # The task says "score is the number of pairs". It must be an integer.
    # (C+N) is always even.
    # C = 2X+Y.
    # If Y=1, C is odd. (C+1)/2 is integer.
    # If Y=0, C is even. (C+0)/2 is integer.
    # So the score is always an integer.
    # So comparing C values is sufficient to find the max score, EXCEPT when C values are equal.
    # If C[N1] == C[N2], then Score[N1] = (C+Y1)/2, Score[N2] = (C+Y2)/2.
    # If Y1=1, Y2=0, then Score[N1] = Score[N2] + 0.5.
    # But Score must be integer. This implies C[N1] and C[N2] cannot be equal if Y differs?
    # Wait.
    # C[N] is the count of ordered pairs.
    # If N is even, N/2 is powerful, then (N/2, N/2) is a pair.
    # C[N] includes this pair once.
    # All other pairs (a,b) with a!=b come in pairs (a,b) and (b,a).
    # So C[N] = 2 * (# distinct a<b) + 1 * (1 if N/2 powerful else 0).
    # So C[N] is odd iff N/2 is powerful.
    # So if C[N1] == C[N2], then Y1 must equal Y2.
    # Proof: C is odd iff Y=1. C is even iff Y=0.
    # If C1 == C2, they have the same parity, so Y1 == Y2.
    # Therefore, Score = (C+Y)/2 is uniquely determined by C.
    # So we can just maximize C[N].
    
    # Great! We just need to find N that maximizes conv[N].
    
    # Find max in conv[2:2*M+1]
    # Note: conv is float. We can just use np.argmax.
    
    # To handle potential floating point noise, we might want to check a few neighbors.
    # But FFT of binary signals is usually very accurate.
    
    # Let's find the index of the maximum.
    # valid_range = conv[2:2*M+1]
    # max_idx_in_slice = np.argmax(valid_range)
    # best_N = max_idx_in_slice + 2
    
    # However, we should verify the count and generate pairs.
    
    # Let's get the top 5 candidates to be safe against numerical errors.
    # k = 5
    # top_k = np.argpartition(valid_range, -k)[-k:]
    
    # Let's just use argmax for simplicity and speed.
    
    # One edge case: What if the max is at the boundaries?
    # N >= 2.
    
    # Execute argmax
    # This is O(N).
    
    # We need to ensure we don't pick N=0 or N=1.
    # Slice starts at 2.
    
    # Let's do it.
    
    # Optimization: Use int64 if possible? No, FFT needs float/complex.
    
    # Get the best N
    # valid_slice = conv[2:2*M+1]
    # best_idx = np.argmax(valid_slice)
    # best_N = best_idx + 2
    
    # But wait, we need to return the pairs.
    # We need to list all pairs (a,b) such that a+b=best_N and a,b powerful.
    # This is O(K) by iterating through powerful_list.
    
    # Let's find the best N first.
    
    # We'll use the top 10 candidates to be safe.
    k_cand = 10
    valid_slice = conv[2:2*M+1]
    
    # If valid_slice is empty? M >= 1, so 2*M+1 >= 3. Slice is non-empty.
    
    # Get indices of top k_cand
    # np.argpartition is not stable, but we just need the values.
    # It returns indices.
    
    # For small k, we can just sort? No, N is large.
    # Use argpartition.
    
    part = np.argpartition(valid_slice, -k_cand)[-k_cand:]
    
    candidates = []
    for idx in part:
        N = idx + 2
        # Calculate exact score
        # C = int(round(valid_slice[idx]))
        # Y = 1 if N % 2 == 0 and (N//2) in powerful_set else 0
        # Score = (C + Y) // 2
        
        # We can just use C as proxy since Y is determined by parity of C.
        # But let's compute exact score to be sure.
        
        C_val = int(round(valid_slice[idx]))
        
        Y_val = 0
        if N % 2 == 0:
            half = N // 2
            if half in powerful_set:
                Y_val = 1
        
        score = (C_val + Y_val) // 2
        
        candidates.append((score, N))
        
    # Sort candidates by score
    candidates.sort(reverse=True)
    
    if not candidates:
        # Fallback
        best_N = 2
        pairs = [[1, 1]] if 1 in powerful_set else []
        return {'N': best_N, 'pairs': pairs}
        
    best_score, best_N = candidates[0]
    
    # Construct pairs for best_N
    pairs = []
    for a in powerful_list:
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set:
            pairs.append([a, b])
            
    # Verify count matches score?
    # len(pairs) should be best_score.
    
    return {'N': best_N, 'pairs': pairs}

