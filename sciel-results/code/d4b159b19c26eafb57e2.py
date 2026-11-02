"""Strategy: Spectral Shaping via Parametric Power Spectrum"""
import numpy as np
import time

N = 600

def compute_score(heights):
    """Compute the normalized maximum of the autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    n = len(heights)
    # Compute autoconvolution via FFT (circular)
    f = np.fft.fft(heights)
    ac = np.fft.ifft(f * np.conj(f)).real
    # Normalization: max(ac) / (sum(heights)^2) * N
    s = np.sum(heights)
    if s == 0:
        return 0.0
    return np.max(ac) * N / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # We parameterize the power spectrum S(k) = |H(k)|^2.
    # We want h = IDFT(sqrt(S) * phase). To keep h real, we need S symmetric and phase 0 or pi?
    # Actually, if we choose S(k) real and non-negative, and set H(k) = sqrt(S(k)) (real), 
    # then h = IDFT(H) will be real if H is symmetric (H[k] = H[N-k]).
    # If H is real and symmetric, h is real and even.
    # We require h >= 0.
    
    # Parametric S(k): 
    # S(k) = 1 + a*cos(2*pi*k/N) + b*cos(4*pi*k/N)
    # We optimize a, b to minimize the score.
    
    best_score = float('inf')
    best_h = np.ones(N) # Fallback uniform
    
    # Initial guess: uniform spectrum (flat S) -> uniform h
    # S(k) = 1 for all k. a=0, b=0.
    # h = IDFT(1) = [N, 0, ...]? No, IDFT of constant is delta?
    # If H(k) = 1, h[0]=1, others 0. Spike.
    # If we want uniform h, we need H(k) to be a delta at 0.
    # So S(k) should be 0 everywhere except k=0.
    # This is hard to parametrize smoothly.
    
    # Alternative: Parameterize h directly in a "smooth" basis?
    # Or just use the spectral shape to create "flat" autocorrelation.
    # If we want C(tau) to be flat, S(k) must be a delta.
    # If S(k) is a delta at k=0, h is constant.
    # So uniform h corresponds to S(k) = delta.
    
    # Let's try to perturb the uniform spectrum.
    # Uniform h: H(k) = N * delta(k). S(k) = N^2 * delta(k).
    # If we add low frequency components to S, h becomes non-uniform.
    
    # Let's define S(k) such that it is peaked at k=0 but with a "tail".
    # S(k) = exp(-alpha * k^2) ? Gaussian in frequency -> Gaussian in time?
    # If S is Gaussian, h is Gaussian.
    # Gaussian h has C(tau) Gaussian. Peak at 0.
    
    # Let's optimize the shape of S using a few coefficients.
    # We will search over (a, b) in S(k) = 1 + a*cos(2*pi*k/N) + b*cos(4*pi*k/N)
    # But we must ensure S(k) >= 0.
    
    def get_h_from_spectrum(a, b):
        k = np.arange(N)
        # Ensure S is non-negative
        S = 1.0 + a * np.cos(2 * np.pi * k / N) + b * np.cos(4 * np.pi * k / N)
        S = np.maximum(S, 0.0)
        
        # H = sqrt(S). To make h real, H must be symmetric.
        # S is symmetric (cosines), so H is symmetric.
        H = np.sqrt(S)
        
        # h = IDFT(H)
        h = np.fft.ifft(H).real
        
        # Enforce non-negativity
        h = np.maximum(h, 0.0)
        
        # Normalize to sum=1 to make score scale invariant check easier?
        # The score function handles scaling.
        # But we should return the raw heights.
        # If h is all zeros, return uniform.
        if np.sum(h) < 1e-9:
            return np.ones(N)
        
        return h

    # Grid search over a, b
    # Range: a, b in [-0.9, 0.9] to keep S positive mostly.
    a_vals = np.linspace(-0.5, 0.5, 20)
    b_vals = np.linspace(-0.5, 0.5, 20)
    
    for a in a_vals:
        if time.time() > deadline:
            break
        for b in b_vals:
            if time.time() > deadline:
                break
            
            h = get_h_from_spectrum(a, b)
            score = compute_score(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
    # If the grid search didn't find anything better than uniform (score 1.0),
    # it's likely the theoretical minimum.
    # Let's also try a random search with more parameters if time permits.
    # But given the stagnation, a structural change like this is good.
    
    # One more check: If the evaluator expects LINEAR convolution, 
    # the score function in the seed uses CIRCULAR (FFT). 
    # I will stick to the seed's score function as the proxy.
    
    return best_h.tolist()

# EVOLVE-BLOCK-END
