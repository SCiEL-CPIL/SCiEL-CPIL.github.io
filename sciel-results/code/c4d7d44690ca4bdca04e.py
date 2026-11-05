import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 20s safety margin
    deadline = start_time + 130.0
    
    def generate_powerful(limit):
        """
        Generate all powerful numbers <= limit.
        Powerful numbers are of the form u^2 * v^3 where gcd(u, v) is not strictly required 
        to be 1 for the set definition, but we can generate them uniquely by iterating 
        cubes and squares.
        Optimization: Precompute squares and cubes.
        """
        pw = set()
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(limit ** (1/3)) + 2
        
        # Precompute cubes
        cubes = []
        for v in range(1, max_v + 1):
            c = v ** 3
            if c > limit:
                break
            cubes.append(c)
            
        # For each cube, find max u such that u^2 * v^3 <= limit
        for c in cubes:
            max_u = isqrt(limit // c)
            # Add all u^2 * c for u in 1..max_u
            # Using a list comprehension and set update is faster than loop+add
            # But we must be careful with memory if max_u is large.
            # For limit=10^8, max_v ~ 464. max_u for c=1 is 10000.
            # Total additions ~ sum_{v=1}^{464} sqrt(10^8/v^3) ~ 10000 * sum v^-1.5 ~ 10000 * 2 ~ 20000.
            # This is very small.
            for u in range(1, max_u + 1):
                pw.add(u * u * c)
        
        return sorted(pw)

    # Target M: We want to go as high as possible.
    # Memory for FFT: 2*M * 4 bytes (float32).
    # M=50M -> 400MB. M=100M -> 800MB.
    # Let's try M=80_000_000 first. If it fails, fallback.
    
    target_M = 80_000_000
    
    # Check time
    if time.time() > deadline - 30:
        target_M = 20_000_000

    try:
        powerful_list = generate_powerful(target_M)
    except Exception:
        # Fallback to smaller M if generation fails (unlikely)
        target_M = 10_000_000
        powerful_list = generate_powerful(target_M)
        
    K = len(powerful_list)
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to numpy array for faster operations if needed, 
    # but we mainly need it for the FFT and verification.
    # powerful_list is sorted.
    
    # Create the indicator array for FFT
    # Size 2*M + 1
    # Use float32 to save memory
    size = 2 * target_M + 1
    arr = np.zeros(size, dtype=np.float32)
    
    # Mark powerful numbers
    # Note: powerful_list contains values <= target_M
    # We place them at their index
    arr[powerful_list] = 1.0
    
    # Perform FFT
    # rfft is efficient for real input
    # We need the linear convolution, so we pad to size 2*M+1 (which is already done by arr size)
    # Actually, np.fft.rfft expects the full array. 
    # The convolution of two signals of length L is length 2L-1.
    # Here, the "signal" is length target_M+1. 
    # So we need FFT size >= 2*target_M+1.
    # Our arr is exactly 2*target_M+1. This is correct for linear convolution 
    # if we zero-pad to at least 2*len(signal)-1. 
    # len(signal) = target_M+1. 2*(target_M+1)-1 = 2*target_M+1. Correct.
    
    fft_arr = np.fft.rfft(arr)
    conv = np.fft.irfft(fft_arr * fft_arr, n=size)
    
    # conv[i] is the number of pairs (a, b) such that a+b=i, where a,b in powerful_set.
    # Since we used the full set, this counts all pairs (a,b) with a,b <= M.
    # Note: This counts ordered pairs (a,b) and (b,a) separately, and (a,a) once.
    # Specifically, conv[i] = sum_{a+b=i} 1.
    # We want distinct pairs a <= b.
    # Let C_total = conv[i].
    # Let D = number of pairs with a != b.
    # Let E = number of pairs with a == b (i.e., 2a=i).
    # C_total = 2*D + E.
    # We want to maximize Pairs = D + E = (C_total + E) / 2.
    # E is 1 if i is even and i/2 is powerful, else 0.
    
    # Find the top candidates
    # We can't just take the max of conv, because we need to adjust for E.
    # However, E is at most 1. So maximizing C_total is a very good proxy.
    # Let's get the top 100 indices by conv value.
    
    # Get indices of top 100
    # argpartition is faster than full sort
    top_k = 100
    # conv is float32, convert to int64 for safety? 
    # Max count is K^2. K~10000 -> 10^8. float32 precision is ~7 decimal digits. 
    # 10^8 is exactly representable? 2^26 ~ 6.7e7. 10^8 is not exact in float32.
    # But the error is small. We will verify exact counts later.
    
    # Find indices where conv is high.
    # Use np.argpartition to get top_k indices
    if len(conv) > top_k:
        top_indices = np.argpartition(conv, -top_k)[-top_k:]
    else:
        top_indices = np.arange(len(conv))
        
    # Sort these top indices by their conv value in descending order
    top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    powerful_set = set(powerful_list)
    
    # Verify top candidates
    for idx in top_indices:
        if time.time() > deadline - 5:
            break
            
        N_cand = int(idx)
        
        # Skip small N that are unlikely to be better if we already have a good count
        # But we must check.
        
        # Calculate exact number of pairs a <= b such that a+b=N_cand and a,b powerful.
        exact_pairs = []
        # Iterate a from 1 to N_cand // 2
        # Since powerful_list is sorted, we can iterate until a > N_cand // 2
        
        # Optimization: 
        # We know conv[N_cand] is high. 
        # We can estimate the max possible count from conv.
        # conv[N_cand] approx 2 * exact_pairs_count (if no a==b) or 2*count - 1.
        
        # Iterate
        # To speed up, we can stop if we find enough pairs? No, we need exact.
        # But we can break early if a > N_cand // 2.
        
        # Use binary search to find the cutoff index in powerful_list?
        # Or just iterate, as K is small (~10k).
        
        count = 0
        for a in powerful_list:
            if a > N_cand // 2:
                break
            b = N_cand - a
            if b in powerful_set:
                count += 1
                exact_pairs.append([a, b])
                
        if count > best_count:
            best_count = count
            best_N = N_cand
            best_pairs = exact_pairs
            
        # If the conv value for this candidate is lower than our current best_count * 2 - 1,
        # and we've processed a few, we might be able to stop?
        # conv[idx] is an upper bound on 2*count (roughly).
        # If conv[idx] < 2 * best_count, then it's impossible for this candidate to beat best_count.
        # Since top_indices is sorted by conv descending, if we hit this condition, we can break.
        if conv[idx] < 2 * best_count:
            break
            
    return {'N': best_N, 'pairs': best_pairs}
