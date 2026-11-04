"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)

def _overlap_vec(h, N):
    """Compute overlap vector for all shifts."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    return np.real(np.fft.ifft(np.conj(H) * C))


def construct():
    N = 100
    
    # Initialize with a cosine wave (structured low-discrepancy-ish)
    # h[i] = 0.5 + A * cos(2*pi*k*i/N)
    # We want sum N/2. Cosine sums to 0 if k not multiple of N.
    # Clip to [0,1].
    h = np.zeros(N)
    for i in range(N):
        h[i] = 0.5 + 0.4 * np.cos(2 * np.pi * 1 * i / N)
    
    # Project to [0,1] and fix sum
    def project(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        for _ in range(100):
            h = np.clip(h, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            if abs(diff) < 1e-12:
                break
        return h
        
    h = project(h)
    
    # Pairwise Coordinate Descent with Ternary Search
    # Objective: Minimize max overlap.
    # For a pair (i, j), we vary h_i -> h_i + delta, h_j -> h_j - delta.
    # This preserves sum.
    
    def evaluate_delta(h, i, j, delta):
        # Create a copy to evaluate overlap
        # Optimization: We only need to update the overlap vector if we do full FFT.
        # For ternary search, we need ~50 evaluations per pair.
        # 100 pairs * 50 evals * 50 passes = 250,000 FFTs of size 100.
        # FFT size 100 is very fast.
        h_new = h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        return _overlap_max(h_new, N)
        
    # To speed up, we can precompute base overlap and update incrementally?
    # No, FFT is fast enough for N=100.
    
    passes = 50
    for p in range(passes):
        # Randomize order to avoid bias
        idx = np.arange(N)
        np.random.shuffle(idx)
        
        for i in idx:
            for j in idx:
                if i == j: continue
                if i > j: continue # Only do i < j to cover all pairs once per pass? 
                # Actually, swapping i,j is symmetric in effect on sum, but delta sign flips.
                # We can just iterate all i != j.
                
                # Determine valid delta range
                # h_i + delta in [0, 1] => -h_i <= delta <= 1 - h_i
                # h_j - delta in [0, 1] => h_j - 1 <= delta <= h_j
                d_min = max(-h[i], h[j] - 1.0)
                d_max = min(1.0 - h[i], h[j])
                
                if d_min >= d_max:
                    continue
                
                # Ternary search for optimal delta in [d_min, d_max]
                a = d_min
                b = d_max
                # Perform ternary search
                # Function is convex (max of linear/quadratic terms)
                for _ in range(30): # 30 iterations is enough for precision
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    f1 = evaluate_delta(h, i, j, m1)
                    f2 = evaluate_delta(h, i, j, m2)
                    if f1 < f2:
                        b = m2
                    else:
                        a = m1
                
                d_opt = (a + b) / 2.0
                h[i] += d_opt
                h[j] -= d_opt
                
                # Ensure numerical stability
                h[i] = np.clip(h[i], 0.0, 1.0)
                h[j] = np.clip(h[j], 0.0, 1.0)
                
    # Final projection
    h = project(h)
    
    return h
# EVOLVE-BLOCK-END
