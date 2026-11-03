import numpy as np
import time
import sys

N = 600
DX = (1.0/4 - (-1.0/4)) / N
X = np.linspace(-1/4, 1/4, N, endpoint=False) + DX/2

def compute_autoconv_max(h):
    """
    Compute the maximum value of the autoconvolution of the step function h.
    h is an array of N non-negative heights.
    """
    # Pad h to next power of 2 for FFT efficiency, or just use N if N is power of 2? 
    # N=600 is not power of 2. Use 1024.
    n_fft = 1024
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    
    # FFT of h
    H = np.fft.rfft(h_padded)
    
    # Autoconvolution in frequency domain
    C_freq = H * np.conj(H)
    
    # Inverse FFT to get time domain autoconvolution
    # Note: np.fft.irfft returns real values
    C = np.fft.irfft(C_freq, n=n_fft)
    
    # The autoconvolution of a function supported on [0, N*DX] is supported on [0, 2*N*DX].
    # We need the maximum over the entire range.
    max_val = np.max(C)
    return max_val

def objective_and_grad(h):
    """
    Objective: max(autoconv) / (integral(h))^2
    """
    h = np.maximum(h, 0) # Enforce non-negativity for evaluation, though bounds handle it in optimizer
    
    integral = np.sum(h) * DX
    if integral < 1e-10:
        return 1e10, np.zeros_like(h)
    
    max_val = compute_autoconv_max(h)
    val = max_val / (integral * integral)
    
    # Gradient approximation or analytical?
    # For L-BFGS-B, we need gradient.
    # Let k be the index of the max in C.
    # d(C_k)/d(h_j) = DX if x_j is within the support of f shifted by x_k.
    # This is complicated to get exactly right with FFT bins.
    # Given time constraints, let's use a numerical gradient or a simpler optimizer.
    # Actually, let's just use Nelder-Mead on a lower-dimensional representation?
    # No, the directive says structural change. Direct optimization is the change.
    # Let's use scipy.optimize.minimize with method='Powell' or 'Nelder-Mead' on a subset?
    # Or just return value for black-box?
    # L-BFGS-B is faster if gradient is available.
    # Let's compute gradient numerically for a few coordinates? Too slow.
    # Let's try a different approach: 
    # The problem is equivalent to finding a probability density p (scaled) that minimizes max(p*p).
    # It is known that the Gaussian minimizes the L-infinity norm of the convolution for a given L1 norm?
    # Actually, the Gaussian is optimal for heat equation spreading, but for autoconvolution?
    # A box function has a triangular autoconvolution. A Gaussian has a Gaussian autoconvolution.
    # Triangular peak height: If f is box of width W and height H, area A=WH.
    # f*f is triangle of base 2W, height H^2 * W.
    # Ratio: (H^2 W) / (WH)^2 = 1/W.
    # To minimize ratio, maximize W. But W is bounded by domain [-1/4, 1/4].
    # So a box filling the domain is a candidate.
    # Seed is box filling domain. Score 1.0.
    # Can we do better?
    # If we use a Gaussian, the peak is lower for the same area?
    # Gaussian: f(x) = H exp(-x^2/sigma^2). Area A = H sigma sqrt(pi).
    # f*f is Gaussian with width sigma*sqrt(2). Peak = H^2 sigma sqrt(pi).
    # Ratio: (H^2 sigma sqrt(pi)) / (H sigma sqrt(pi))^2 = 1 / (sigma sqrt(pi)).
    # To minimize ratio, maximize sigma.
    # But the function must be supported on [-1/4, 1/4].
    # If we truncate the Gaussian, the area decreases.
    # Let's compare Box vs Gaussian with same support width.
    # Box: W=0.5. Ratio = 1/0.5 = 2. (Wait, my previous ratio calc: max = H^2 W. Area = HW. Ratio = H^2 W / H^2 W^2 = 1/W. Yes.)
    # Gaussian truncated to [-L, L]:
    # This is complex.
    # However, it is known that for fixed L1 norm and support, the function that minimizes the L-infinity norm of the convolution is related to the "flat" function?
    # No, actually, spreading the mass out more reduces the peak of the convolution.
    # The box function spreads mass uniformly.
    # What if we use a "cosine" bump? f(x) = 0.5(1+cos(pi x / L)) on [-L, L].
    # This is smoother.
    
    # Let's use a parametric search for a Cosine Bump or a "Super-Gaussian" exp(-|x|^p).
    # This is a structural change from the seed (Box) to a parametric family of smooth bumps.
    # But inspiration 1/2/3 already did parametric/Fourier.
    # The directive says "MATERIALLY DIFFERENT".
    # The seed is a flat box.
    # If I use a smooth bump, I am in the same family as inspirations 1-3.
    # Is there a different family?
    # What about a **two-box** or **multi-box** structure?
    # Or a **random** structure optimized by ES?
    
    # Let's try **Evolutionary Strategy** on the raw 600-dimensional vector.
    # This is a structural change: from deterministic parametric to stochastic global search on high-dim space.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initial guess: A Gaussian-like bump centered at 0
    # Support is [-0.25, 0.25].
    # Let's try a "Super-Gaussian" exp(-|x|^p) with p=4.
    # Or just start with the Box and do local search?
    # If I start with Box, and do local search, I might stay in the "Box" basin.
    # Let's start with a Gaussian.
    
    # Generate initial Gaussian
    sigma = 0.15
    h_init = np.exp(-(X**2) / (2*sigma**2))
    h_init = np.maximum(h_init, 0)
    
    # Normalize to have integral 1? The ratio is scale invariant, so normalization doesn't matter for the score, 
    # but it helps numerical stability.
    integral = np.sum(h_init) * DX
    h_init = h_init / integral # Now integral is 1.
    
    best_h = h_init
    best_val = compute_autoconv_max(best_h) / (np.sum(best_h)*DX)**2
    
    # Use scipy.optimize.minimize with L-BFGS-B
    # We need a gradient. Let's use a finite difference gradient for L-BFGS-B? 
    # That's N evaluations per gradient step. Too slow.
    # Let's use Nelder-Mead on a reduced set of parameters?
    # Or use `scipy.optimize.minimize` with `method='Powell'` which is derivative-free.
    # Powell on 600 variables might be slow, but let's try with maxiter limit.
    
    from scipy.optimize import minimize
    
    def obj(h):
        h = np.maximum(h, 0)
        integral = np.sum(h) * DX
        if integral < 1e-10:
            return 1e10
        max_val = compute_autoconv_max(h)
        return max_val / (integral * integral)
    
    # Bounds: [0, inf]
    bounds = [(0, None)] * N
    
    # Try Powell first, it's robust for derivative-free.
    # But 600 dims is a lot for Powell.
    # Alternative: Coordinate Descent.
    
    # Let's implement a simple Coordinate Descent with Line Search.
    # For each coordinate i, find the best value for h[i] while keeping others fixed.
    # This is O(N * cost_eval). Cost_eval is O(N log N). Total O(N^2 log N).
    # 600^2 * 10 = 3.6M ops per sweep. Very fast.
    # We can do many sweeps.
    
    h = best_h.copy()
    
    for sweep in range(200): # Max 200 sweeps
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Randomize order to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            
            old_hi = h[i]
            
            # We want to minimize obj(h). 
            # Fix all other h[j].
            # The objective is max(C) / I^2.
            # I changes with h[i]. C changes with h[i].
            # Let's do a small line search around old_hi.
            
            # Try a few candidates
            # 0, old_hi, old_hi * 1.1, old_hi * 0.9, etc.
            # Better: Use a small window.
            # Since the function is non-negative, and we are minimizing,
            # we can try to adjust h[i].
            
            # Let's try 5 points in a small range around old_hi
            # Range: [0, max(0.1, old_hi * 1.5)]
            if old_hi == 0:
                candidates = [0.0, 1e-4, 1e-3, 1e-2]
            else:
                # Relative perturbation
                candidates = [
                    max(0, old_hi * 0.5),
                    max(0, old_hi * 0.8),
                    old_hi,
                    old_hi * 1.2,
                    old_hi * 1.5,
                    0.0
                ]
            
            best_candidate_val = obj(h)
            best_candidate_h = old_hi
            
            for cand in candidates:
                h[i] = cand
                val = obj(h)
                if val < best_candidate_val:
                    best_candidate_val = val
                    best_candidate_h = cand
            
            h[i] = best_candidate_h
            
            if abs(best_candidate_h - old_hi) > 1e-6:
                improved = True
            
            # Update best
            if obj(h) < best_val:
                best_val = obj(h)
                best_h = h.copy()
        
        if not improved:
            # Add some noise to escape local minima
            noise_scale = 0.01 * np.mean(h)
            h = h + np.random.normal(0, noise_scale, N)
            h = np.maximum(h, 0)
            # Re-normalize? No, scale invariant.
            # But keep it roughly same scale
            integral = np.sum(h) * DX
            if integral > 1e-10:
                h = h / integral

    # Final refinement with a tighter local search on best_h
    # Use Nelder-Mead on a subset of the largest elements?
    # Or just return best_h
    
    return list(best_h)

# Test locally? No, just output.
# One issue: obj(h) calls compute_autoconv_max which uses FFT.
# 200 sweeps * 600 coords * 6 candidates = 720,000 evaluations.
# Each eval is FFT(1024). 1024 FFT is ~10us.
# 720,000 * 10us = 7.2 seconds.
# This is well within 120s.
# We can increase sweeps or candidates.

# Let's refine the candidates to be more adaptive.
# And ensure we don't waste time on zero-height bins if they stay zero.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Initial guess: Gaussian
    sigma = 0.15
    h_init = np.exp(-(X**2) / (2*sigma**2))
    h_init = np.maximum(h_init, 0)
    integral = np.sum(h_init) * DX
    if integral > 1e-10:
        h_init = h_init / integral
    else:
        h_init = np.ones(N) / N
    
    best_h = h_init
    best_val = obj(best_h)
    
    h = best_h.copy()
    
    # Adaptive Coordinate Descent
    for sweep in range(500):
        if time.time() > deadline - 5:
            break
        
        improved = False
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            
            old_hi = h[i]
            
            # If height is very small, try to turn it on
            # If height is large, try to turn it down
            # Use a log-spaced grid of candidates?
            
            if old_hi < 1e-5:
                # Try to activate
                candidates = [0.0, 1e-4, 1e-3, 1e-2, 0.01]
            else:
                # Local refinement
                # Use relative steps
                step = 0.1 * old_hi
                candidates = [
                    max(0, old_hi - 2*step),
                    max(0, old_hi - step),
                    old_hi,
                    old_hi + step,
                    old_hi + 2*step,
                    0.0
                ]
            
            current_val = obj(h)
            best_candidate_val = current_val
            best_candidate_h = old_hi
            
            for cand in candidates:
                if cand == old_hi:
                    continue
                h[i] = cand
                val = obj(h)
                if val < best_candidate_val:
                    best_candidate_val = val
                    best_candidate_h = cand
            
            h[i] = best_candidate_h
            
            if abs(best_candidate_h - old_hi) > 1e-6:
                improved = True
            
            if obj(h) < best_val:
                best_val = obj(h)
                best_h = h.copy()
        
        if not improved:
            # Perturb
            noise_scale = 0.05 * np.mean(h[h>0]) if np.any(h>0) else 0.01
            h = h + np.random.normal(0, noise_scale, N)
            h = np.maximum(h, 0)
            integral = np.sum(h) * DX
            if integral > 1e-10:
                h = h / integral

    return list(best_h)

# Note: obj is defined inside construct in the thought process, 
# but I need to define it outside or inside properly.
# I defined `obj` inside `construct` in the thought, but `compute_autoconv_max` is global.
# Let's move `obj` to global for clarity and speed (avoiding closure overhead? negligible).

def obj(h):
    h = np.maximum(h, 0)
    integral = np.sum(h) * DX
    if integral < 1e-10:
        return 1e10
    max_val = compute_autoconv_max(h)
    return max_val / (integral * integral)

# Final check on `construct` signature and return type.
# Returns list of floats.
