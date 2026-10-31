import numpy as np
from scipy import optimize

N = 600

def _fft_convolve(h):
    """Compute linear autoconvolution using FFT for speed."""
    # Pad to a power of 2 for speed, or use next_fast_len if available
    # 1200 is 2N. 2048 is the next power of 2.
    fft_size = 2048
    H = np.fft.rfft(h, fft_size)
    P = H * np.conj(H)
    conv = np.fft.irfft(P, fft_size)
    # The linear convolution is in the first 2N-1 bins
    return conv[:2*N-1]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, leaving 20s margin for safety
    
    # Initial guess: all ones
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_autoconv_max(h)
    
    # Soft-max objective for smooth optimization
    # Lower T is better for approximation, but harder to optimize.
    # We will use adaptive T or a low T with many iterations.
    def soft_objective(x, T=0.01):
        h = np.abs(x)
        conv = _fft_convolve(h)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv_max = np.max(conv)
        # Log-sum-exp for stability
        # Shift by max to avoid overflow
        shifted = (conv - conv_max) / T
        # Clip to avoid exp under/overflow
        shifted = np.clip(shifted, -50, 50)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    # Exact objective for final evaluation
    def exact_objective(x):
        h = np.abs(x)
        return compute_autoconv_max(h)
    
    # Strategy: Multiple restarts with diverse initializations
    # 1. Uniform
    # 2. Random
    # 3. Frequency-domain "flattened" spectrum
    # 4. Alternating signs (projected to positive)
    
    np.random.seed(42)
    
    # Generate diverse initial guesses
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Random positive
    inits.append(np.random.rand(N) + 0.1)
    
    # 3. Frequency domain flat spectrum
    # Try to create a signal with flat power spectrum
    # H(k) = exp(i * phi_k) with random phases
    # h = IFFT(H)
    # We need h >= 0. This is hard to guarantee directly.
    # Instead, let's take a random real signal, take FFT, flatten magnitude, keep phase, IFFT, rectify.
    h_rand = np.random.randn(N)
    H_rand = np.fft.rfft(h_rand, 2048)
    # Flatten magnitude
    mag = np.ones_like(np.abs(H_rand))
    phase = np.angle(H_rand)
    H_flat = mag * np.exp(1j * phase)
    h_flat = np.fft.irfft(H_flat, 2048)[:N]
    h_flat = np.abs(h_flat)
    if np.max(h_flat) > 0:
        h_flat = h_flat / np.max(h_flat)
    inits.append(h_flat)
    
    # 4. Cosine window (reduces spectral leakage, might help)
    t = np.linspace(0, 1, N, endpoint=False)
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * t))
    inits.append(h_cos)
    
    # 5. Random phase Barker-like (just random)
    inits.append(np.random.rand(N))
    
    # 6. Two-tone
    h_two = 1.0 + 0.5 * np.cos(2 * np.pi * 5 * t)
    h_two = np.maximum(h_two, 0.1)
    inits.append(h_two)
    
    # 7. High frequency noise
    inits.append(1.0 + 0.2 * np.random.randn(N))
    inits[-1] = np.maximum(inits[-1], 0.1)
    
    # 8. Edge enhanced
    h_edge = 1.0 + 0.5 * np.abs(np.linspace(-1, 1, N))
    inits.append(h_edge)
    
    # 9. Center enhanced
    h_center = 1.0 + 0.5 * np.exp(-5 * (np.linspace(-1, 1, N))**2)
    inits.append(h_center)
    
    # 10. Random with seed 1
    np.random.seed(1)
    inits.append(np.random.rand(N) + 0.1)
    
    # 11. Random with seed 2
    np.random.seed(2)
    inits.append(np.random.rand(N) + 0.1)
    
    # 12. Random with seed 3
    np.random.seed(3)
    inits.append(np.random.rand(N) + 0.1)
    
    # Optimization Loop
    # We have ~130s. 
    # FFT conv is fast. L-BFGS is the bottleneck.
    # Let's do 2 stages: Coarse (T=0.1, 100 iter) and Fine (T=0.01, 200 iter)
    # Or just one stage with moderate T and many iterations?
    # L-BFGS converges fast. 
    # Let's try 12 restarts, each with 100 iterations of L-BFGS with T=0.05.
    # Then polish the best ones with T=0.01.
    
    candidates = []
    
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        # Normalize h0 to have sum 1 for consistent scale? 
        # The objective is scale invariant: max(conv)/(sum^2). 
        # Scaling h by c scales conv by c^2 and sum^2 by c^2. Ratio is invariant.
        # So we don't need to normalize, but it helps numerical stability.
        h0 = h0 / np.sum(h0) * N # Average value 1
        
        try:
            # Stage 1: Coarse optimization
            result1 = optimize.minimize(
                soft_objective, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if time.time() > deadline - 20:
                break
                
            candidates.append(result1.x)
            
        except Exception:
            pass
    
    # Evaluate all candidates with exact objective
    for x in candidates:
        h_new = np.abs(x)
        score = compute_autoconv_max(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
            
    # Polish the best solution
    if time.time() < deadline - 10:
        try:
            # Use a lower temperature for the final polish
            def soft_objective_fine(x, T=0.005):
                h = np.abs(x)
                conv = _fft_convolve(h)
                total = np.sum(h)
                if total < 1e-10:
                    return 1e10
                conv_max = np.max(conv)
                shifted = (conv - conv_max) / T
                shifted = np.clip(shifted, -50, 50)
                soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
                return soft_max / (total * total)
            
            result_polish = optimize.minimize(
                soft_objective_fine, best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            h_new = np.abs(result_polish.x)
            score = compute_autoconv_max(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
                
        except Exception:
            pass
            
    return list(best_h)
