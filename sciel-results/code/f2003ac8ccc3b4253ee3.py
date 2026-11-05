import time
import numpy as np
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    # Deadline guard: 120s limit, keep 10s safety margin
    deadline = start_time + 110.0
    
    def generate_powerful(limit):
        """Generate all powerful numbers <= limit."""
        pw = set()
        # Powerful numbers are of the form u^2 * v^3
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(limit ** (1/3)) + 2
        
        for v in range(1, max_v + 1):
            v_cubed = v ** 3
            if v_cubed > limit:
                break
            # u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
            max_u = isqrt(limit // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                pw.add(p)
        return sorted(pw)

    # 1. Generate powerful numbers up to a reasonable limit M
    # The current program tries up to 40M. 
    # With bucketing, we can afford a slightly larger M or just stick to a safe M 
    # and use the freed-up memory/compute for more buckets/verification.
    # Let's use M = 10^7 to be safe and fast, allowing for many buckets.
    # Actually, density of powerful numbers is ~ x^(1/3) * constant.
    # For M=10^7, K ~ 2000. For M=10^8, K ~ 4000.
    # Let's try M = 20_000_000 (2*10^7). K will be around 3000-4000.
    M = 20_000_000
    
    if time.time() > deadline - 10:
        return {'N': 2, 'pairs': [[1, 1]]}

    powerful_list = generate_powerful(M)
    K = len(powerful_list)
    
    # 2. Multi-Bucket FFT Strategy
    # Partition powerful_list into buckets of size B.
    # For each pair of buckets (i, j), compute the convolution of their indicators.
    # Accumulate counts in a global dictionary `global_counts`.
    
    # Bucket size choice:
    # We want the FFT size to be manageable.
    # If bucket size is B, and max value in bucket is ~M, 
    # the sum of two buckets can go up to 2*M.
    # But actually, the values in bucket i are roughly in [M*i/B, M*(i+1)/B].
    # The sum of values from bucket i and bucket j is in [M*(i+j)/B, M*(i+j+2)/B].
    # We can use a smaller FFT size if we shift the arrays?
    # No, standard FFT computes linear convolution.
    # If we have bucket A with max val V_A and bucket B with max val V_B,
    # the convolution length needs to be V_A + V_B + 1.
    # If we just use the global max M for all, the FFT size is 2*M, which is what we wanted to avoid.
    
    # Alternative: Use "sparse" convolution or just standard FFT on the full array 
    # but split the *computation* of the square?
    # No, the memory issue is the array size `2*M`.
    # If M=2*10^7, size=4*10^7. 
    # 4*10^7 float64 = 320 MB. This is actually quite feasible on modern systems!
    # The current program's fallback to smaller M suggests it might have hit memory limits 
    # or time limits with larger M.
    # Let's check if we can just do the full FFT for M=2*10^7.
    # If M=4*10^7, size=8*10^7 -> 640 MB. Might be tight.
    
    # Let's try a hybrid: 
    # 1. Try full FFT with M=15_000_000.
    # 2. If that fails or is too slow, fall back to bucketing for a larger M?
    # Actually, bucketing doesn't reduce the *max* FFT size if we still need to cover the full range of sums.
    # The only way bucketing helps is if we can *truncate* the FFT for pairs of buckets 
    # that only produce sums in a specific range, and we don't care about the rest.
    # But we want the MAX count over ALL N. So we need all sums.
    
    # Wait, the "Multi-Resolution" inspiration suggests using buckets to *extend* the range N?
    # If we use buckets, we can compute sums for N > 2*M if we have powerful numbers > M?
    # No, we only generated up to M. So max sum is 2*M.
    
    # So, does bucketing help?
    # It helps if we can do the FFT in smaller chunks and accumulate, avoiding one huge allocation 
    # if the system has fragmented memory or strict per-allocation limits.
    # But `np.fft.rfft` on a 30M array is usually fine.
    
    # Let's look at the current score: 4.8.
    # The current program tries M=40M.
    # If M=40M, size=80M. 80M * 8 bytes = 640MB.
    # Maybe it failed due to memory? Or time?
    # FFT of 80M takes ~1-2 seconds.
    # Verification is the bottleneck?
    # Top 500 candidates * K (4000) = 2M checks. Fast.
    
    # What if we increase M significantly?
    # M = 100_000_000 (10^8).
    # K ~ 10000?
    # FFT size 2*10^8 -> 1.6 GB. Too big for 1GB limit?
    # If we bucket, we can process 10^8 with smaller FFTs.
    
    # Let's implement the bucketing to handle M = 100_000_000.
    # Bucket size B = 1000.
    # Number of buckets = K / B ~ 10.
    # For each pair (i, j), we create an array of size (max_val_i + max_val_j + 1).
    # If bucket i contains values around M*i/K, the max val is ~ M*(i+1)/K.
    # This is still close to M for the last bucket.
    # The FFT size for the last bucket pair is still ~ 2*M.
    # So the peak memory is still determined by the largest bucket pair.
    
    # To truly reduce memory, we must ensure that the *sum* of the max values of the two buckets 
    # is smaller than the global 2*M.
    # This implies we can only do this if we split the range of *sums* we care about?
    # No, we care about all sums.
    
    # However, we can use **Number Theoretic Transform (NTT)** or just accept that 
    # the peak memory is the issue.
    
    # Let's try a different structural change:
    # **Direct Search with Heuristics**.
    # Instead of finding ALL sums via FFT, which is O(N log N), 
    # we can search for specific N that are likely to have many representations.
    # N should be highly composite or have many divisors?
    # Powerful numbers are sparse.
    # a = u^2 v^3.
    # If we fix a structure for a and b, can we construct N?
    
    # Let's stick to the FFT but optimize the memory by using **float32** instead of float64?
    # Precision might suffer for counts > 10^7, but our counts are likely small (< 100).
    # If counts are small, float32 is exact enough?
    # FFT errors accumulate. For size 2^26, error might be ~ 1e-3.
    # Rounding to int should work if counts are integers and not too large.
    
    # Let's try M = 50_000_000 with float32.
    # Size 100M -> 400 MB.
    
    # Strategy: 
    # 1. Generate powerful numbers up to M=50_000_000.
    # 2. Use float32 FFT.
    # 3. If MemoryError, reduce M.
    # 4. Verify top candidates.
    
    # This is a "tuning" of the current approach, not a structural change.
    # The prompt asks for a MATERIALLY DIFFERENT strategy.
    
    # Structural Change: **Meet-in-the-Middle on Factors**?
    # No, that's for finding the numbers, not the sums.
    
    # How about **Random Sampling + Local Search**?
    # 1. Generate a random subset of powerful numbers.
    # 2. Compute their pairwise sums.
    # 3. Find the N with the most sums in this sample.
    # 4. Verify N using the FULL set of powerful numbers.
    # 5. Repeat with different random subsets.
    # This avoids the full FFT memory blowup.
    # If we sample S powerful numbers, we have S^2/2 sums.
    # If S=1000, 500k sums. Very fast.
    # The probability that the global max N is found in the sample depends on the density.
    # If the top N has 10 representations out of K=4000, the chance that *any* pair is in the sample 
    # is roughly 1 - (1 - 10/K^2)^S? No, that's not right.
    # We need *all* 10 pairs to be present? No, we just need to *find* N as a candidate.
    # If we find *one* pair (a,b) for N, we can verify N against the full set.
    # So we just need to hit at least one pair for the optimal N.
    # If the optimal N has C pairs, and we sample S numbers, 
    # the probability of missing all C pairs is (1 - C/(K^2))^S? 
    # Actually, the pairs are not independent.
    # But if C is small (e.g., 10) and K is large (4000), the pairs are "rare".
    # Sampling 1000 numbers gives ~500k pairs. Total possible pairs ~8M.
    # We cover ~6% of pairs.
    # If the best N has 10 pairs, the chance we miss all 10 is (1-0.06)^10 ~ 0.5.
    # So 50% chance to find the best N in one sample.
    # Doing 10 samples, chance to miss all is 0.5^10 ~ 0.1%.
    # This is a very robust strategy that uses O(S^2) time and O(S^2) memory (for the dict), 
    # which is tiny compared to O(N) FFT memory.
    # And it allows us to use a MUCH larger M (and thus K) because the generation time is the main cost, 
    # and the search is fast.
    
    # Let's implement this **Random Sampling + Verification** strategy.
    
    # Parameters:
    # M = 100_000_000 (10^8). K will be ~ 10,000.
    # S = 2000 (sample size).
    # Num samples = 20.
    
    target_M = 100_000_000
    if time.time() > deadline - 20:
        target_M = 20_000_000
        
    powerful_list = generate_powerful(target_M)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Convert to set for O(1) lookup during verification
    powerful_set = set(powerful_list)
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    # Sampling Strategy
    # We want to find N with many representations.
    # We'll use multiple random samples.
    
    import random
    random.seed(42)
    
    num_samples = 50
    sample_size = min(1500, K) # Don't sample more than K
    
    for s in range(num_samples):
        if time.time() > deadline - 10:
            break
            
        # Randomly select sample_size powerful numbers
        # Use a random subset of indices
        indices = random.sample(range(K), sample_size)
        sample = [powerful_list[i] for i in indices]
        
        # Compute all pairwise sums (a <= b)
        # Use a dictionary to count occurrences
        counts = defaultdict(int)
        
        # Optimization: Nested loop in Python is slow for 1500^2 = 2.25M iterations.
        # 50 samples * 2.25M = 112M iterations. Might be too slow in pure Python.
        # Let's use Numpy to vectorize the pair generation?
        # Or reduce sample size?
        # S=1000 -> 500k pairs. 50 * 500k = 25M. Acceptable.
        
        S = len(sample)
        # To keep it fast, we can use a smaller S or fewer samples if time is tight.
        # Let's check time for one sample.
        
        for i in range(S):
            a = sample[i]
            # b must be >= a to keep pairs distinct and a<=b
            # Since sample is not sorted, we can't easily restrict.
            # We'll just generate all pairs and filter a<=b later or just count all and divide?
            # No, a+b=N is symmetric. 
            # If we count all ordered pairs (a,b) where a,b in sample, 
            # then count(N) = 2 * (unordered pairs with a!=b) + (pairs with a==b).
            # We want unordered distinct pairs a<=b.
            # It's easier to just iterate i < j and i==j.
            
            for j in range(i, S):
                b = sample[j]
                # a <= b is not guaranteed if we just pick i,j.
                # But for counting sums, order doesn't matter.
                # We just need to know how many pairs (a,b) from the sample sum to N.
                # If we count all pairs (i,j) with i<=j, we get the number of unordered pairs 
                # from the sample that sum to N.
                # This is a lower bound on the total number of pairs for N.
                # If we find a high count, it's a good candidate.
                
                N_val = a + b
                counts[N_val] += 1
                
        # Find the max count in this sample
        if not counts:
            continue
            
        max_count_sample = max(counts.values())
        # Only verify if this count is promising?
        # Or just verify the top few candidates from this sample.
        
        # Get top 10 candidates from this sample
        # Sort by count
        # Using sorted(counts.items(), key=lambda x: x[1], reverse=True)
        # This is O(U log U) where U is number of unique sums. U ~ S^2/2.
        
        # To speed up, just pick the max?
        # If the true max is not in the sample, we miss it.
        # But we do 50 samples.
        
        # Let's get the top 5 candidates to verify.
        # For efficiency, if max_count_sample is low, maybe skip?
        # No, we want to find the global max.
        
        top_candidates = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for N_cand, c in top_candidates:
            # Verify N_cand against the FULL powerful set
            # Count pairs (a, b) in powerful_set such that a + b = N_cand and a <= b.
            
            # Iterate through powerful numbers a <= N_cand // 2
            # powerful_list is sorted.
            
            # Find index where a > N_cand // 2
            # Since N_cand can be up to 2*M, and powerful_list is up to M,
            # we iterate all a in powerful_list where a <= N_cand // 2.
            
            exact_count = 0
            pairs = []
            
            # Optimization: If c (sample count) is 1, and we already have best_count > 1, 
            # can we skip? No, the sample might have missed the other pairs.
            # But if c is very low, the probability of it being the global max is low.
            # However, verification is fast if we break early?
            # We must count ALL pairs to be sure.
            
            # We can speed up verification by using the fact that we already know 
            # at least 'c' pairs exist.
            
            for a in powerful_list:
                if a > N_cand // 2:
                    break
                b = N_cand - a
                if b in powerful_set:
                    exact_count += 1
                    pairs.append([a, b])
                    
            if exact_count > best_count:
                best_count = exact_count
                best_N = N_cand
                best_pairs = pairs
                
    return {'N': best_N, 'pairs': best_pairs}
