"""
Strategy: Structured Exhaustive Search + Scaling
The score S(U) = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1) is homogeneous under scaling.
If V = k*U, then S(V) = S(U).
Thus, we can find the best "shape" (small integers) and then scale it to fit the time budget
or simply return the small shape if scaling doesn't help (it doesn't change score, but 
larger max(U) might introduce numerical issues or just waste time; actually, 
the score is identical for U and kU. So we just need the best small U).

However, the prompt says "maximize the sumset/difference-set exponent lower bound".
The score formula is fixed. 
S(U) = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)

Let's analyze small sets.
For a set U, let N=|U|.
|U+U| <= N(N+1)/2.
|U-U| <= N^2.
Ratio R = |U-U|/|U+U|.
Max possible R is roughly 2 (if U+U is small, U-U is large).
Actually, for symmetric sets around 0 (shifted), |U-U| is symmetric.
For non-negative sets containing 0:
U-U contains negative, 0, positive.
U+U contains only non-negative.
|U-U| is typically larger than |U+U| because of the negative part.
Specifically, if U is "sum-free" (no a+b=c), |U+U| is large (~N^2/2).
If U is "difference-free" (no a-b=c), |U-U| is large.
We want Large |U-U| and Small |U+U|.

Small sets analysis:
U = {0, a, b}
Sums: 0, a, b, 2a, a+b, 2b. (6 if distinct)
Diffs: 0, a, b, -a, -b, a-b, b-a, 2a, -2a, 2b, -2b. 
Unique diffs: 0, +/-a, +/-b, +/-a-b, +/-2a, +/-2b.
If a, b are such that a-b != 0, a, b, 2a, 2b...
Let's just brute force small N.

Why this is different:
Current: Greedy local search starting from geometric.
New: Exhaustive search over all subsets of {1..K} of size N, for N in [3..8], K in [10..30].
This is a different design family (Exhaustive/Brute Force over small domain vs Greedy Heuristic).
Since N is small, we can enumerate C(K, N) which is manageable.
C(20, 6) ~ 38k. C(30, 8) ~ 5.8M (too slow in Python).
C(15, 5) ~ 3000. C(20, 4) ~ 4845.
We can afford C(25, 6) ~ 177k.
With fast score computation (vectorized numpy), 177k * ~0.1ms = 17s. Feasible.

We will search for the best small set U_small.
Then we return U_small. (Scaling doesn't change score, so returning the small one is valid and fast).
Wait, does scaling change the score?
S(kU) = 1 + log(|kU-kU|/|kU+kU|) / log(2*k*max(U)+1)
|kU-kU| = |U-U| (scaled by k, but cardinality is same)
|kU+kU| = |U+U| (cardinality is same)
So numerator is same.
Denominator is log(2k*max(U)+1).
As k increases, denominator increases, so S(kU) DECREASES.
So we want max(U) to be as SMALL as possible!
Therefore, we just need to find the best set of small integers.
No scaling needed. In fact, scaling is bad.

So the task reduces to: Find U subset of {0, 1, ..., K} that maximizes S(U).
Since S(U) decreases with max(U) for a fixed ratio, and ratio depends on structure,
we should search small K.

Let's search N up to 10, K up to 20.
C(20, 10) is 184,756.
Score computation needs to be very fast.
Using FFT for sumset is overkill for small K.
Bitsets are faster.
Represent U as a bitmask or boolean array.
Sumset: Convolution. For small K, simple O(N^2) set union is fine.
N=10, N^2=100. 184k * 100 = 18M ops. Python sets are slow.
Vectorized numpy:
Create boolean array A of size 2*maxU+1.
Convolve A with A.
Count non-zeros.
This is fast.

Plan:
1. Iterate N from 3 to 9.
2. Iterate K from N to 25.
3. Generate all combinations of size N-1 from {1..K} (0 is fixed).
4. For each combination, compute score.
5. Keep best.
6. Return best set.
"""

import time
import math
import itertools
import numpy as np

def compute_score(U):
    """
    Computes the score for a given set U (list of non-negative ints).
    Uses vectorized numpy for speed.
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Sumset |U+U|
    # Max sum is 2*maxU
    # Create array A where A[u] = 1 if u in U
    # Size needs to be at least 2*maxU + 1 for linear convolution to fit without wrap-around
    # We use rfft which is efficient.
    max_sum = 2 * maxU
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    A_fft = np.fft.rfft(A)
    C_fft = A_fft * A_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    
    # Sumset size is count of non-zeros in C[0..2*maxU]
    # Threshold 0.5 to avoid floating point noise
    sum_set_size = np.sum(C[:max_sum + 1] > 0.5)
    
    # Diffset |U-U|
    # Range [-maxU, maxU]
    # We can compute this using the same convolution trick?
    # U-U = U + (-U).
    # Let B be array where B[maxU - u] = 1 for u in U.
    # Then conv(A, B) gives counts of u1 + (maxU - u2) = maxU + (u1 - u2).
    # So index k in conv corresponds to diff d = k - maxU.
    # Range of k: 0..2*maxU.
    # Range of d: -maxU..maxU.
    
    B = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        B[maxU - u] = 1.0
        
    B_fft = np.fft.rfft(B)
    D_fft = A_fft * B_fft
    D = np.fft.irfft(D_fft, n=n_fft)
    
    # Diffset size is count of non-zeros in D[0..2*maxU]
    # Note: D[k] > 0 implies maxU + (u1 - u2) = k => u1 - u2 = k - maxU.
    # So all diffs are covered.
    diff_set_size = np.sum(D[:max_sum + 1] > 0.5)
    
    if sum_set_size == 0:
        return 0.0
        
    log_ratio = math.log(diff_set_size / sum_set_size)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0.0
    best_U = [0, 1]
    
    # We search for sets of size N containing 0.
    # So we choose N-1 elements from {1, ..., K}.
    # Since score decreases with max(U), we should prioritize small K.
    # However, a larger K might allow a better ratio structure that outweighs the log penalty.
    # Let's search N from 3 to 10.
    # For each N, search K from N to 25.
    
    # Optimization: Precompute FFTs? No, U changes.
    # But we can skip K if time is tight.
    
    max_N = 10
    max_K = 25
    
    # Sort search order: Smaller K first? Or iterate N and K.
    # Let's iterate K from 1 to max_K, and for each K, iterate N.
    # This allows us to find good small sets early.
    
    # Actually, generating combinations is the bottleneck.
    # C(25, 9) is 2M. Too slow.
    # C(20, 7) is 77k.
    # C(25, 5) is 53k.
    
    # Strategy:
    # 1. Try small N (3, 4, 5) with larger K (up to 30).
    # 2. Try medium N (6, 7) with smaller K (up to 20).
    
    search_space = []
    for N in range(3, max_N + 1):
        # Heuristic: Limit K based on N to keep C(K, N-1) manageable
        # C(K, N-1) < 100k
        # Approx K ~ 20 for N=7
        # Approx K ~ 15 for N=8
        max_K_for_N = int(35 - 2.5 * N) 
        if max_K_for_N < N:
            max_K_for_N = N
        if max_K_for_N > 30:
            max_K_for_N = 30
            
        for K in range(N, max_K_for_N + 1):
            search_space.append((N, K))
            
    # Sort by expected cost? Or just run in order.
    # Let's run smaller K first to get a baseline quickly?
    # Or smaller N first?
    # Smaller N has fewer combinations.
    
    for N, K in search_space:
        if time.time() > deadline - 10:
            break
            
        # Generate combinations of size N-1 from range(1, K+1)
        # 0 is always in the set.
        base = [0]
        
        # Optimization: If K is small, generate all.
        # If K is large, we might timeout.
        # Let's estimate count.
        # math.comb(K, N-1)
        try:
            count = math.comb(K, N-1)
        except:
            count = 10**9
            
        if count > 200000:
            # Skip or sample? Skip to save time for other configs.
            continue
            
        for comb in itertools.combinations(range(1, K + 1), N - 1):
            if time.time() > deadline - 2:
                break
                
            U = base + list(comb)
            # U is sorted because comb is sorted and base is [0]
            # But comb starts from 1, so U is sorted.
            
            s = compute_score(U)
            
            if s > best_score:
                best_score = s
                best_U = U
                
    # If we found nothing (shouldn't happen), fallback
    if not best_U:
        best_U = [0, 1, 3]
        
    return best_U

# EVOLVE-BLOCK-END
