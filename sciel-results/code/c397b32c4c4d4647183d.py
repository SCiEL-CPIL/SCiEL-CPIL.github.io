"""Second autocorrelation inequality: choose 50 nonnegative heights to maximize
the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution.

Strategy:
- The score is (||h||_2^2) / (||h||_1 * ||h * h||_inf) where * is convolution.
- Since h is nonnegative, ||h||_1 = sum(h), ||h||_2^2 = sum(h^2), and (h*h)[k] = sum_i h[i]*h[k-i].
- The supremum of the autoconvolution is at least sum(h)^2 / N if spread uniformly, but can be larger.
- For a single spike: ||h||_2^2 = 1, ||h||_1 = 1, (h*h)[0] = 1, so ratio = 1/1 = 1... wait.
  Actually if h = [0,...,1,0,...], then ||h||_2^2 = 1, ||h||_1 = 1, (h*h) has one entry = 1, so ratio = 1/(1*1) = 1.
- For uniform h = [1,1,...,1]: ||h||_2^2 = N, ||h||_1 = N, (h*h)[k] = N-|k| (for discrete convolution of length N with itself).
  The max of (h*h) is at k=0: N. So ratio = N / (N * N) = 1/N = 1/50 = 0.02.
  
Wait, let me reconsider. The task says "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."

So score = ||h||_2^2 / (||h||_1 * ||h*h||_inf)

For a single spike: ||h||_2^2 = 1, ||h||_1 = 1, ||h*h||_inf = 1. Score = 1/1 = 1.
For uniform: ||h||_2^2 = N, ||h||_1 = N, ||h*h||_inf = N. Score = N/(N*N) = 1/N = 0.02.

So the spike is much better! The seed program (uniform) gives 1.0 score (per the contract, the seed scores 1.0).

Wait, re-reading: "score is the lower bound on C2" and "the search score is value/seed, so 1.0 equals the seed program."

So the actual metric value for the seed (uniform) is some number, and the score reported is value/seed_value. The seed gives score 1.0.

Let me think about what the actual value is. If score = ||h||_2^2 / (||h||_1 * ||h*h||_inf), then:
- Uniform: 50 / (50 * 50) = 0.02
- Spike: 1 / (1 * 1) = 1

So the spike would give score = 1/0.02 = 50 relative to uniform. But the observation says "Concentrating mass in a single spike gives ratio ~100 vs uniform's ~2." This suggests the metric might be different, or there's a normalization.

Actually, looking more carefully: "Measured: 1 -> 1.003" means a program that returned the spike got score 1.003. So the spike barely beats the seed.

Let me reconsider the metric. Perhaps it's:
score = ||h||_2^2 / (||h||_1^2 * ||h*h||_inf / ||h||_1^2) = ||h||_2^2 / ||h*h||_inf

No, let me re-read: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"

This could be: (||h||_2^2) / (||h||_1 * ||h*h||_inf)

Or it could be: ||h||_2^2 / (||h||_1 * ||h*h||_inf) where the autoconvolution is normalized somehow.

Given that the spike scores 1.003 and uniform scores 1.0, the difference is small. Let me think about what metric would give these values.

If we normalize by ||h||_1, let g = h/||h||_1 so ||g||_1 = 1. Then:
||h||_2^2 = ||h||_1^2 * ||g||_2^2
||h*h||_inf = ||h||_1^2 * ||g*g||_inf

So score = ||h||_1^2 * ||g||_2^2 / (||h||_1 * ||h||_1^2 * ||g*g||_inf) = ||g||_2^2 / (||h||_1 * ||g*g||_inf)

That still depends on ||h||_1 unless there's a constraint.

Perhaps the heights are constrained to sum to 1, or the max is 1, or something. Let me just implement the metric as described and do local optimization.

The key insight from the observation: "Concentrating mass in a single spike gives ratio ~100 vs uniform's ~2" - this suggests the raw values are ~100 and ~2, and the ratio between them is 50x, but the score (relative to seed) is 1.003 for the spike. This is confusing.

Let me just implement the score function as ||h||_2^2 / (||h||_1 * ||h*h||_inf) and do local optimization starting from a spike.
"""

import numpy as np
import time

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    # h has length N, autoconvolution has length 2N-1
    # Use FFT for efficiency
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Start with a single spike
    h = np.zeros(N)
    h[N // 2] = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Try various initial configurations
    configs = [
        np.zeros(N),
    ]
    # Single spike at center
    c = np.zeros(N); c[N//2] = 1.0; configs.append(c)
    # Single spike at various positions
    for i in range(N):
        c = np.zeros(N); c[i] = 1.0; configs.append(c)
    # Two adjacent spikes
    for i in range(N - 1):
        c = np.zeros(N); c[i] = 1.0; c[i+1] = 1.0; configs.append(c)
    # Block of k consecutive 1s
    for k in range(2, 15):
        c = np.zeros(N); c[:k] = 1.0; configs.append(c)
        c = np.zeros(N); c[N-k:] = 1.0; configs.append(c)
    # Centered block
    for k in range(2, 15):
        start = (N - k) // 2
        c = np.zeros(N); c[start:start+k] = 1.0; configs.append(c)
    # Geometric decay
    for r in [0.5, 0.7, 0.8, 0.9, 0.95]:
        c = np.array([r**i for i in range(N)]); configs.append(c)
        c = np.array([r**(N-1-i) for i in range(N)]); configs.append(c)
    # Symmetric geometric
    for r in [0.7, 0.8, 0.9, 0.95]:
        idx = np.arange(N)
        center = (N - 1) / 2.0
        c = r ** (np.abs(idx - center))
        configs.append(c)
    
    for c in configs:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_h = c.copy()
        if time.time() > deadline - 10:
            break
    
    # Local optimization: perturb individual bins
    current_h = best_h.copy()
    current_score = best_score
    
    # Try random restarts with local optimization
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Initialize: either from best or random
        if restart % 3 == 0:
            h = best_h.copy()
        elif restart % 3 == 1:
            h = np.random.exponential(1.0, N)
            if np.sum(h) > 0:
                h /= np.sum(h)
                h *= np.random.uniform(1, 5)
        else:
            h = np.zeros(N)
            k = np.random.randint(1, 10)
            idx = np.random.choice(N, k, replace=False)
            h[idx] = 1.0
        
        h = np.maximum(h, 0)
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # Local optimization steps
        steps = 200
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Perturb a random subset of bins
            n_perturb = np.random.randint(1, 6)
            indices = np.random.choice(N, n_perturb, replace=False)
            h_new = h.copy()
            for idx in indices:
                if np.random.random() < 0.5:
                    # Add small noise
                    h_new[idx] += np.random.normal(0, 0.1)
                else:
                    # Scale
                    h_new[idx] *= np.random.uniform(0.5, 1.5)
                h_new[idx] = max(0, h_new[idx])
            
            s_new = _score(h_new)
            if s_new > s:
                h = h_new
                s = s_new
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
        
        # Also try coordinate descent on individual bins
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            idx = np.random.randint(N)
            old_val = h[idx]
            h[idx] = max(0, h[idx] * np.random.uniform(0.5, 1.5))
            s_new = _score(h)
            if s_new > s:
                s = s_new
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
            else:
                h[idx] = old_val
    
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    uniform = np.ones(N)
    print("Uniform score:", _score(uniform))
    
    spike = np.zeros(N)
    spike[N//2] = 1.0
    print("Spike score:", _score(spike))
    
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
