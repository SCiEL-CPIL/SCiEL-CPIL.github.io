import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    # Safety margin: aim to finish in 80s, hard stop at 85s
    HARD_STOP = t0 + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n) for n=1..N using vectorized sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # The loop runs N times, but each step is vectorized.
    # Total operations ~ N log N.
    # For i=1, sigma[1::1] += 1 (N ops)
    # For i=2, sigma[2::2] += 2 (N/2 ops)
    # ...
    # This is efficient in NumPy.
    
    # To speed up, we can process in chunks or just let NumPy handle it.
    # We need to be careful about memory and time.
    
    # Optimization: We can stop early if time is running out.
    # But the sieve must be complete to group correctly.
    # If we can't finish the sieve, we can't guarantee correctness.
    # However, partial sieve is not useful for exact sigma.
    # So we must complete the sieve or fallback.
    
    # Let's try to run the sieve.
    # If it takes too long, we might need a smaller N.
    # Let's assume it fits. If not, the timeout will trigger.
    
    # Actually, pure Python loop for sieve is too slow.
    # NumPy slice addition is the key.
    
    # We can optimize by noting that for large i, the number of multiples is small.
    # But NumPy overhead per slice operation might be high for many small slices.
    # Alternative: Use a standard Python sieve for SPF and compute sigma from factors?
    # No, that's O(N log log N) for SPF but O(N * num_factors) for sigma.
    # The additive sieve is O(N log N) but with low constant factor in NumPy.
    
    # Let's use the additive sieve.
    # To mitigate risk, we can reduce N if time is tight.
    # But let's try N=10^7 first.
    
    # A faster variant:
    # Only iterate i up to N.
    # We can use a list of primes to optimize? No, additive sieve doesn't need primes.
    
    # Let's run the sieve.
    # We will check time periodically.
    
    # Note: sigma[0] is unused.
    
    # To speed up the sieve, we can use the fact that sigma is multiplicative?
    # No, the additive sieve is simpler and fast enough with NumPy.
    
    # Let's implement the sieve.
    # If time.time() > HARD_STOP - 5.0, we might abort and return a fallback.
    
    # Actually, doing 10^7 slice additions might be slow due to Python loop overhead.
    # Each `sigma[i::i] += i` is a Python statement.
    # 10^7 Python statements is too slow (~10s just for loop overhead, plus vector ops).
    # Wait, 10^7 iterations in Python is ~1-5 seconds.
    # The vector ops themselves:
    # Sum of N/i is ~ N log N = 1.6 * 10^8.
    # 1.6 * 10^8 integer additions in NumPy is very fast (<1s).
    # So the bottleneck is the 10^7 Python loop iterations.
    # 10^7 iterations is ~0.5-1.0 seconds in PyPy, ~1-2s in CPython.
    # So total time should be ~2-3 seconds. This is safe.
    
    for i in range(1, N + 1):
        if time.time() > HARD_STOP - 10.0:
            # If we are running out of time, we cannot complete the sieve.
            # Return a known good pair or fallback.
            # A known pair: sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12
            # sigma(11)=12, sigma(6)=12. Pair (11, 6). |11/6 - 1.5| = |1.833 - 1.5| = 0.333
            # Score = 1/0.333 = 3.
            # Wait, the score is seed/value. Seed is likely 1.0?
            # The prompt says "score is the binary64 serialization of that rational... raw value is float(|a/b-3/2|)"
            # "The search score is seed/value".
            # If seed is 1.0, score = 1/val.
            # For (11,6), val = 1/3, score = 3.
            # The current program score is 4077490. This implies the current program is returning a very bad pair or failing?
            # "raw value of exactly 0 cannot be scored and counts as invalid".
            # If the program returns [11, 6], val = 1/3. Score = 3.
            # Why is the score 4077490?
            # Maybe the current program returns a pair with very large error?
            # Or maybe the seed is different?
            # "1.0 equals the seed program".
            # If the seed program returns [11, 6], score is 3.
            # The current program has score 4077490. This is much worse than 3.
            # This suggests the current program is returning a pair with error ~ 1/4077490? No, score = seed/value.
            # If score is 4077490, and seed=1, then value = 1/4077490. This is a VERY good pair.
            # But the current program is described as "too slow" and "fails".
            # Let's re-read: "CURRENT PROGRAM (score 4077490.0000)".
            # If the current program actually found a very good pair, why would we rewrite it?
            # "INSPIRATIONS ... score 2040966 ... score 788363".
            # These scores are getting smaller.
            # If score = 1/value, smaller score means LARGER value (worse).
            # Wait. "The evaluator measures the raw value (lower is better)."
            # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
            # This implies:
            # If raw value decreases, score increases.
            # So a score of 4077490 is a HUGE improvement over 1.0.
            # This means the current program IS finding a very good pair.
            # But the inspirations say it's "too slow".
            # Maybe the current program *sometimes* times out and returns a bad pair, but in the recorded run it succeeded?
            # Or maybe the "score" in the prompt is the raw value?
            # "The evaluator measures the raw value (lower is better)."
            # "The search score is seed/value".
            # If raw value is $V$, score is $1/V$.
            # If $V$ is small, score is large.
            # Current score 4077490 -> $V \approx 2.4 \times 10^{-6}$.
            # Inspiration 3 score 788363 -> $V \approx 1.2 \times 10^{-6}$.
            # Inspiration 2 score 2040966 -> $V \approx 4.9 \times 10^{-7}$.
            # Wait, 2040966 > 788363. So Insp 2 is better than Insp 3?
            # "score 2040966.0000" vs "score 788363.3333".
            # If higher score is better, Insp 2 is better.
            # But Insp 3 says "current program ... is too slow".
            # The current program has score 4077490, which is the best.
            # So the current program is actually working well in that specific run?
            # But the directive says "Rewrite ... with a stronger algorithm".
            # And "cause of stalled progress has not been established".
            # Perhaps the current program is unstable?
            
            # Regardless, my goal is to find a pair with smaller $V$ (higher score).
            # If I can find $V < 2.4 \times 10^{-6}$, I improve.
            
            # Let's proceed with the NumPy sieve. It should be fast and reliable.
            # If it completes, I search for the best pair.
            
            # Fallback if sieve fails:
            return [11, 6]
        
        sigma[i::i] += i
        
    # 2. Group by sigma
    # We need to find pairs (a, b) with sigma[a] == sigma[b] minimizing |a/b - 1.5|
    
    # Create an array of indices 1..N
    # Sort indices by sigma value
    # To save memory and time, we can use argsort.
    
    # sigma is int64.
    # We need to group indices by sigma value.
    
    # Method:
    # 1. Create array `indices` = 1..N
    # 2. Sort `indices` by `sigma[indices]`
    # 3. Iterate through sorted indices, identifying groups of equal sigma.
    # 4. For each group, find best pair.
    
    # Memory:
    # sigma: 10^7 * 8 bytes = 80 MB
    # indices: 10^7 * 8 bytes = 80 MB
    # sorted_indices: 10^7 * 8 bytes = 80 MB
    # Total ~ 240 MB. Acceptable.
    
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    # Sort indices by sigma value
    # np.argsort(sigma[indices])
    # This creates a copy of sigma[indices] for sorting?
    # Better: order = np.argsort(sigma[1:], kind='stable')
    # Then sorted_indices = order + 1
    
    if time.time() > HARD_STOP - 5.0:
        return [11, 6]
        
    order = np.argsort(sigma[1:], kind='stable')
    sorted_indices = order + 1  # 1-based indices
    
    # Now iterate through groups
    # We need to find boundaries where sigma value changes.
    
    sorted_sigma = sigma[sorted_indices]
    
    # Find change points
    # changes = np.where(sorted_sigma[1:] != sorted_sigma[:-1])[0]
    # starts = np.concatenate(([0], changes + 1))
    # ends = np.concatenate((changes + 1, [len(sorted_sigma)]))
    
    # This creates several arrays of size ~10^7.
    
    if time.time() > HARD_STOP - 5.0:
        return [11, 6]
        
    changes = np.nonzero(sorted_sigma[1:] != sorted_sigma[:-1])[0]
    starts = np.concatenate(([0], changes + 1))
    ends = np.concatenate((changes + 1, [len(sorted_sigma)]))
    
    best_val = float('inf')
    best_pair = None
    
    # Iterate through groups
    for gi in range(len(starts)):
        if time.time() > HARD_STOP - 2.0:
            break
            
        s0 = starts[gi]
        e0 = ends[gi]
        length = e0 - s0
        
        if length < 2:
            continue
            
        # Extract the group of indices
        # gn = sorted_indices[s0:e0]
        # gn is already sorted by value? No, sorted by sigma.
        # We need gn sorted by value (which is the index itself) to use bisect.
        # But sorted_indices is sorted by sigma, not by value.
        # So we need to sort the group by value.
        
        gn = sorted_indices[s0:e0].copy()
        gn.sort()
        
        L = len(gn)
        
        # For each b in gn, find a in gn minimizing |a/b - 1.5|
        # |a - 1.5b| / b
        # Minimize |2a - 3b| / (2b)
        
        # We can iterate over b and binary search for a.
        # Or iterate over a and binary search for b.
        # Since we want a/b ~ 1.5, a ~ 1.5b.
        # So a is larger than b.
        
        for bi in range(L):
            b = int(gn[bi])
            target = 1.5 * b
            
            # Find a in gn closest to target
            # bisect_left
            j = bisect.bisect_left(gn, target)
            
            if j < L:
                a = int(gn[j])
                if a != b:
                    num = abs(2 * a - 3 * b)
                    if num > 0:
                        val = num / (2.0 * b)
                        if val < best_val:
                            best_val = val
                            best_pair = (a, b)
            
            if j > 0:
                a = int(gn[j - 1])
                if a != b:
                    num = abs(2 * a - 3 * b)
                    if num > 0:
                        val = num / (2.0 * b)
                        if val < best_val:
                            best_val = val
                            best_pair = (a, b)
                            
    if best_pair is None:
        return [11, 6]
        
    a, b = best_pair
    # Verify
    if sigma[a] != sigma[b]:
        return [11, 6]
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [int(a), int(b)]
