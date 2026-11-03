import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    dw = 2.0 / N
    
    def compute_overlaps_fft(h):
        """
        Compute overlap at all N circular shifts using FFT.
        
        overlap_k = dw * sum_i min(h[i], 1 - h[(i+k) % N])
        
        We can write:
        min(a, b) = (a + b - |a - b|) / 2
        
        So overlap_k = dw * sum_i [h[i] + (1 - h[(i+k)%N]) - |h[i] - (1 - h[(i+k)%N])|] / 2
                     = dw * sum_i [h[i] + 1 - h[(i+k)%N] - |h[i] - 1 + h[(i+k)%N]|] / 2
        
        Let's expand:
        sum_i h[i] = S = N/2
        sum_i 1 = N
        sum_i h[(i+k)%N] = S = N/2
        
        So the first three terms sum to: dw * (N/2 + N - N/2) / 2 = dw * N / 2
        
        The fourth term: -dw/2 * sum_i |h[i] - 1 + h[(i+k)%N]|
        
        Let g[i] = h[i] - 1, so g[i] + h[(i+k)%N] = h[i] - 1 + h[(i+k)%N]
        
        We need sum_i |g[i] + h[(i+k)%N]| for all k.
        This is the cross-correlation of |.| with the signal, but |.| is not linear.
        
        Alternative approach: use the identity
        min(a,b) = (a+b-|a-b|)/2
        
        overlap_k = dw/2 * sum_i [h[i] + 1 - h[(i+k)%N] - |h[i] - 1 + h[(i+k)%N]|]
        
        = dw/2 * [N/2 + N - N/2 - sum_i |h[i] + h[(i+k)%N] - 1|]
        
        = dw/2 * [N - sum_i |h[i] + h[(i+k)%N] - 1|]
        
        So we need to compute A_k = sum_i |h[i] + h[(i+k)%N] - 1| for all k.
        
        This is a circular cross-correlation with a non-linear function.
        We can compute it directly in O(N^2) which is 10000 operations, fast enough.
        Or we can use FFT if we can express it, but the absolute value makes it tricky.
        
        For N=100, O(N^2) = 10000 is fine.
        """
        overlaps = np.zeros(N)
        for k in range(N):
            h_shifted = np.roll(h, -k)
            # min(h[i], 1 - h_shifted[i])
            overlaps[k] = np.sum(np.minimum(h, 1.0 - h_shifted)) * dw
        return overlaps
    
    def compute_overlaps_vectorized(h):
        """Vectorized computation of all overlaps using broadcasting (O(N^2) memory but fast)."""
        # h[i] and h[(i+k)%N] for all i, k
        # Create matrix M[i,k] = h[(i+k)%N]
        idx = np.arange(N)
        M = h[(idx[:, None] + idx[None, :]) % N]  # M[i,k] = h[(i+k)%N]
        # min(h[i], 1 - M[i,k])
        mins = np.minimum(h[:, None], 1.0 - M)
        overlaps = np.sum(mins, axis=0) * dw
        return overlaps
    
    # The analytical solution: h_i = 0.5 for all i
    # This gives overlap_k = dw * sum_i min(0.5, 0.5) = dw * N * 0.5 = (2/N) * N/2 = 1.0
    # Wait: dw * N * 0.5 = (2/N) * N * 0.5 = 1.0
    # And the lower bound is N/4 * dw = (N/4) * (2/N) = 0.5? Let me recheck.
    
    # Actually, let me recompute. overlap_k = dw * sum_i min(h[i], 1-h[(i+k)%N])
    # For h[i] = 0.5: min(0.5, 0.5) = 0.5
    # sum_i 0.5 = N/2
    # overlap_k = dw * N/2 = (2/N) * N/2 = 1.0
    
    # The task says "score is the upper bound on C5" and "the evaluator measures the raw value (lower is better)"
    # With h=0.5, max overlap = 1.0, so score = 1.0.
    
    # Let me verify the lower bound. 
    # Sum_k overlap_k = dw * sum_k sum_i min(h[i], 1-h[(i+k)%N])
    # For each i, as k ranges over 0..N-1, (i+k)%N ranges over all j.
    # So sum_k min(h[i], 1-h[(i+k)%N]) = sum_j min(h[i], 1-h[j])
    # Sum_k overlap_k = dw * sum_i sum_j min(h[i], 1-h[j])
    
    # min(a,b) = (a+b-|a-b|)/2
    # sum_i sum_j min(h[i], 1-h[j]) = sum_i sum_j (h[i] + 1 - h[j] - |h[i] - 1 + h[j]|)/2
    # = (1/2) * [sum_i sum_j h[i] + sum_i sum_j 1 - sum_i sum_j h[j] - sum_i sum_j |h[i] + h[j] - 1|]
    # = (1/2) * [N * S + N^2 - N * S - sum_i sum_j |h[i] + h[j] - 1|]
    # = (1/2) * [N^2 - sum_i sum_j |h[i] + h[j] - 1|]
    
    # where S = sum h[i] = N/2
    
    # So sum_k overlap_k = dw/2 * [N^2 - sum_i sum_j |h[i] + h[j] - 1|]
    # = (2/N)/2 * [N^2 - sum_i sum_j |h[i] + h[j] - 1|]
    # = (1/N) * [N^2 - sum_i sum_j |h[i] + h[j] - 1|]
    # = N - (1/N) * sum_i sum_j |h[i] + h[j] - 1|
    
    # Average overlap = (1/N) * sum_k overlap_k = 1 - (1/N^2) * sum_i sum_j |h[i] + h[j] - 1|
    
    # To minimize max overlap, we want to minimize the average, which means maximizing sum_i sum_j |h[i] + h[j] - 1|.
    
    # For h[i] = 0.5: |0.5 + 0.5 - 1| = 0, so sum = 0, average overlap = 1.
    
    # Can we do better? If we make h more extreme (closer to 0 or 1), we get larger |h[i]+h[j]-1|.
    # But we're constrained to h[i] in [0,1] and sum = N/2.
    
    # Let's try a binary-like solution: half 0s and half 1s.
    # h[i] = 1 for i < N/2, h[i] = 0 for i >= N/2.
    # sum = N/2. Good.
    # overlap_k = dw * sum_i min(h[i], 1-h[(i+k)%N])
    # For k=0: min(1,0)=0 for i<N/2, min(0,1)=0 for i>=N/2. overlap_0 = 0.
    # For k=N/2: h[(i+N/2)%N] is 0 for i<N/2, 1 for i>=N/2.
    #   min(1, 1-0)=1 for i<N/2, min(0, 1-1)=0 for i>=N/2.
    #   overlap_{N/2} = dw * N/2 * 1 = (2/N) * N/2 = 1.
    # So max overlap is at least 1. Same as uniform.
    
    # Hmm. Let me try a different pattern.
    # What about h[i] = 1 if i mod 2 == 0, 0 otherwise? (for N=100)
    # sum = 50. Good.
    # overlap_k = dw * sum_i min(h[i], 1-h[(i+k)%N])
    # For k even: h[(i+k)%N] has same parity as h[i], so h[(i+k)%N] = h[i].
    #   min(h[i], 1-h[i]): if h[i]=1, min(1,0)=0; if h[i]=0, min(0,1)=0. overlap_k = 0.
    # For k odd: h[(i+k)%N] = 1-h[i].
    #   min(h[i], 1-(1-h[i])) = min(h[i], h[i]) = h[i].
    #   overlap_k = dw * sum_i h[i] = dw * N/2 = 1.
    # So max overlap = 1. Again same.
    
    # It seems hard to beat 1.0. The uniform solution achieves 1.0, and the lower bound 
    # (average) is 1 - (1/N^2)*sum|...|. For uniform, sum|...| = 0, so average = 1.
    # For any other h, sum|...| > 0, so average < 1, but max >= average.
    # The question is whether we can make max < 1.
    
    # Actually wait: if average < 1, it's possible that max < 1 if all overlaps are below 1.
    # But can we achieve that?
    
    # Let me try a small example. N=4, h = [0, 1, 0, 1], sum=2=N/2.
    # dw = 2/4 = 0.5.
    # overlap_0: min(0,1)+min(1,0)+min(0,1)+min(1,0) = 0+0+0+0=0. overlap_0=0.
    # overlap_1: h_shifted = [1,0,1,0]. min(0,0)+min(1,1)+min(0,0)+min(1,1)=0+1+0+1=2. overlap_1=0.5*2=1.
    # overlap_2: h_shifted = [0,1,0,1]. Same as overlap_0. overlap_2=0.
    # overlap_3: h_shifted = [1,0,1,0]. Same as overlap_1. overlap_3=1.
    # Max = 1.
    
    # N=4, h=[0.5,0.5,0.5,0.5]: all overlaps = 0.5*4*0.5=1. Max=1.
    
    # N=4, h=[0,0,1,1]: 
    # overlap_0: min(0,1)+min(0,1)+min(1,0)+min(1,0)=0. 
    # overlap_1: h_shifted=[0,1,1,0]. min(0,1)+min(0,0)+min(1,0)+min(1,1)=0+0+0+1=1. overlap_1=0.5.
    # overlap_2: h_shifted=[1,1,0,0]. min(0,0)+min(0,0)+min(1,1)+min(1,1)=0+0+1+1=2. overlap_2=1.
    # overlap_3: h_shifted=[1,0,0,1]. min(0,0)+min(0,1)+min(1,1)+min(1,0)=0+0+1+0=1. overlap_3=0.5.
    # Max=1.
    
    # It seems 1.0 might be the optimal value. Let me just return the uniform solution.
    
    # But wait, the task says "seed/value" where 1.0 equals the seed program. The seed likely returns h=0.5.
    # So the best we can do is 1.0.
    
    # Let me just verify with a quick optimization to see if we can beat 1.0.
    
    best_h = np.full(N, 0.5)
    best_score = compute_overlaps_vectorized(best_h).max()
    
    # Try a few random patterns
    rng = np.random.default_rng(42)
    for trial in range(200):
        if time.time() > DEADLINE:
            break
        # Random pattern with sum = N/2
        h = rng.random(N)
        h = np.clip(h, 0, 1)
        target = N / 2.0
        current = h.sum()
        if current > 0:
            h = h * (target / current)
            h = np.clip(h, 0, 1)
            diff = target - h.sum()
            if abs(diff) > 1e-10:
                mask = (h > 0) & (h < 1)
                if mask.sum() > 0:
                    h[mask] += diff / mask.sum()
                    h = np.clip(h, 0, 1)
        
        score = compute_overlaps_vectorized(h).max()
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Return the best found
    # Ensure exact sum
    target = N / 2.0
    diff = target - best_h.sum()
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-15) & (best_h < 1 - 1e-15)
        if mask.sum() > 0:
            best_h[mask] += diff / mask.sum()
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
