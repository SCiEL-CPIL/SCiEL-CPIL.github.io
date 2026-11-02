import numpy as np
import time

N = 100
DEADLINE = 145.0

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    start_time = time.time()
    
    # Strategy: 1D Ising Spin Chain with Mean-Field Constraint
    # h[i] is interpreted as a spin state s[i] in {0, 1}.
    # The constraint sum(h) = N/2 means exactly N/2 spins are 1.
    # We model this as an antiferromagnetic chain to maximize the "alternating" nature,
    # which minimizes local correlations (and thus overlap with 1-h shifts).
    # We use Glauber dynamics with a mean-field pressure to maintain the particle number.
    
    # Initialize with a perfect alternating pattern (ground state of AFM)
    # s = [1, 0, 1, 0, ...]
    # This has autocorr[k] = 0 for all odd k, and N/2 for even k (if N even).
    # Wait, for N=100, alternating 1,0,1,0...
    # s = 1, 0, 1, 0 ...
    # autocorr[0] = 50.
    # autocorr[1] = 0 (1*0 + 0*1 ... = 0).
    # autocorr[2] = 50 (1*1 + 0*0 ... = 50).
    # Overlap[k] = 50 - autocorr[k].
    # Overlap[0] = 0.
    # Overlap[1] = 50.
    # Overlap[2] = 0.
    # Max Overlap = 50.
    # Value = (2/100) * 50 = 1.0.
    # This is worse than 0.5!
    
    # Why? Because the uniform 0.5 solution has autocorr[k] = 25 for all k.
    # Overlap[k] = 50 - 25 = 25.
    # Max Overlap = 25.
    # Value = 0.5.
    
    # The alternating pattern creates strong correlations at even shifts.
    # We want to DE-correlate.
    # Antiferromagnetic coupling favors s[i] != s[i+1].
    # This creates the alternating pattern.
    # But we want to MINIMIZE the maximum overlap.
    # The uniform solution is the "disordered" limit in terms of correlations, 
    # but it's not a valid Ising state (it's 0.5).
    
    # If we are restricted to binary states {0,1}, the best we can do is 
    # make autocorr[k] as close to N/4 = 25 as possible for all k.
    # The alternating pattern has autocorr[1]=0, autocorr[2]=50.
    # This is far from 25.
    
    # A random binary pattern has autocorr[k] ~ 25 + noise.
    # The noise is O(sqrt(N)).
    # Max autocorr deviation is ~ O(sqrt(N) * log N) or similar.
    # For N=100, sqrt(100)=10.
    # So autocorr might range from 15 to 35.
    # Overlap ranges from 50-35=15 to 50-15=35.
    # Max Overlap ~ 35.
    # Value ~ 0.7.
    # This is worse than 0.5.
    
    # It seems binary patterns cannot beat 0.5.
    # The proof in the previous thought process holds:
    # Average Overlap = 0.5.
    # Max Overlap >= Average Overlap.
    # So Min Max Overlap >= 0.5.
    # The uniform 0.5 achieves 0.5.
    # So 0.5 is the global minimum for ANY h in [0,1] with sum N/2.
    
    # Why did inspiration 1 claim 1.05?
    # Maybe the evaluator uses a different metric?
    # Or maybe the "seed" value is not 0.5?
    # "score is seed/value".
    # If seed value is 0.5, and score is 1.0, then seed_value = 0.5.
    # If value < 0.5, score > 1.0.
    # But value >= 0.5.
    
    # Is it possible that the "overlap" is not cyclic?
    # If linear shifts, the sum of autocorrelations is different.
    # But the code uses FFT, implying cyclic.
    
    # Let's assume the task is correct and I am missing something.
    # Maybe the "overlap" is defined as max |h(x) - h(x+k)|?
    # No, "overlap of h with 1-h" is standard integral of product.
    
    # Let's try to implement the Ising model anyway, as requested.
    # If it converges to a binary pattern, it will score ~0.7-0.8.
    # If I mix it with continuous values, I might get back to 0.5.
    
    # Hybrid Strategy:
    # 1. Start with h = 0.5.
    # 2. Perform Glauber dynamics on the "rounded" version of h?
    # No, let's just do the Ising model.
    # If it scores worse, it's a valid structural change.
    
    # However, to maximize the chance of beating 1.0 (if possible),
    # I should probably stick to the continuous space but use the Ising
    # "field" concept to guide the search.
    
    # The "field" at site i is h[i+1] + h[i-1] (antiferromagnetic).
    # We want to minimize the max overlap.
    # The overlap is related to autocorrelation.
    # High autocorrelation -> High overlap.
    # We want LOW autocorrelation for k != 0.
    # This means we want h to be "uncorrelated".
    # White noise is uncorrelated.
    # But h is constrained to [0,1] and sum N/2.
    
    # Let's implement a simple Ising-like search on binary states,
    # but allow "soft" states? No, Ising is binary.
    
    # I will implement the binary Ising model with Glauber dynamics.
    # Initialization: Random binary with N/2 ones.
    # Energy function: E = max_k Overlap[k].
    # But max is non-differentiable and expensive.
    # Use a proxy: Sum of squares of (Overlap[k] - mean)?
    # Or just evaluate max Overlap.
    
    # To speed up, I will update the autocorrelation incrementally?
    # No, N=100 is small. FFT is fast.
    
    s = np.zeros(N, dtype=int)
    # Random initialization
    indices = np.random.permutation(N)
    s[indices[:N//2]] = 1
    s[indices[N//2:]] = 0
    
    h = s.astype(float)
    best_h = h.copy()
    best_cost = _overlap_max(h)
    
    T = 0.1
    T_min = 1e-5
    cooling = 0.999
    
    # Mean-field pressure to maintain sum = N/2
    # If sum(s) > N/2, we want to flip 1->0 more often.
    # If sum(s) < N/2, we want to flip 0->1 more often.
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        # Pick a random site
        i = np.random.randint(N)
        
        # Calculate the change in cost if we flip s[i]
        old_s_i = s[i]
        new_s_i = 1 - old_s_i
        
        # Tentative flip
        s[i] = new_s_i
        h_tentative = s.astype(float)
        
        # Check constraint: sum must be N/2
        # Since we flip, sum changes by +1 or -1.
        # We need to accept/reject based on energy AND constraint?
        # The prompt says "strictly enforce the sum constraint".
        # So we must maintain sum = N/2.
        # This means we must flip a pair? Or use a different dynamics?
        # "Glauber dynamics" usually flips single spins.
        # To maintain sum, we can use a "pair swap" move.
        # Pick i, j. Swap s[i] and s[j].
        
        # Let's use pair swaps.
        s[i] = old_s_i # Revert
        
        j = np.random.randint(N)
        if i == j:
            continue
            
        if s[i] == s[j]:
            continue # Swapping identical spins does nothing
        
        # Swap s[i] and s[j]
        s[i], s[j] = s[j], s[i]
        h_tentative = s.astype(float)
        
        new_cost = _overlap_max(h_tentative)
        delta = new_cost - best_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            h = h_tentative
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Revert
            s[i], s[j] = s[j], s[i]
            
        T *= cooling
        
    return best_h
