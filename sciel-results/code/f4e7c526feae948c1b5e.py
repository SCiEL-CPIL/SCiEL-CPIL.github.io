import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return 0.0
    return np.max(conv) / (total * total)

def soft_objective(x, T=0.01):
    """Smooth soft-max objective for L-BFGS-B optimization."""
    h = np.maximum(x, 0)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -100, 100)
    soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
    return soft_max / (total * total)

def _flat_spectrum_init(seed, n_modes=5, amplitude=0.15):
    """Generate initialization by targeting a flat power spectrum.
    
    Start from a random phase spectrum, compute the inverse FFT,
    and take the absolute value to get a nonnegative function
    with approximately flat power spectrum.
    """
    rng = np.random.RandomState(seed)
    
    # Random phases for a flat power spectrum
    phases = rng.uniform(0, 2 * np.pi, N)
    
    # Add some low-frequency structure for diversity
    if n_modes > 0:
        # Pick n_modes random frequencies
        freqs = rng.choice(N, size=n_modes, replace=False)
        for f in freqs:
            phases[f] = rng.uniform(0, 2 * np.pi)
            if f < N - 1:
                phases[N - f] = -phases[f]  # Ensure real signal
    
    # Complex spectrum with unit magnitude and random phases
    H = np.exp(1j * phases)
    
    # Inverse FFT to get time-domain signal
    h = np.real(ifft(H, N))
    
    # Make nonnegative and normalize
    h = np.abs(h)
    if np.max(h) > 0:
        h = h / np.max(h)
    
    # Mix with uniform for stability
    h = (1 - amplitude) * np.ones(N) + amplitude * h
    h = np.maximum(h, 0.01)
    
    return h

def _cosine_init(seed, n_harmonics=3):
    """Generate initialization using cosine harmonics."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, N, endpoint=False)
    
    h = np.ones(N)
    for k in range(1, n_harmonics + 1):
        freq = rng.uniform(1, 8)
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.05 / k
        h += amp * np.cos(2 * np.pi * freq * t + phase)
    
    h = np.maximum(h, 0.01)
    return h

def _edge_enhanced_init(seed):
    """Enhance edges to reduce central peak."""
    rng = np.random.RandomState(seed)
    t = np.linspace(-1, 1, N)
    h = 1.0 + 0.3 * np.abs(t) + 0.05 * rng.randn(N)
    h = np.maximum(h, 0.01)
    return h

def _random_init(seed):
    """Pure random initialization."""
    rng = np.random.RandomState(seed)
    h = rng.rand(N) + 0.1
    h = h / np.mean(h)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Initial guess: all ones
    h = np.ones(N)
    best_score = compute_autoconv_max(h)
    best_h = h.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # Flat spectrum inits (frequency-domain insight)
    for seed in [42, 123, 456, 789, 1000, 2000]:
        initializations.append(_flat_spectrum_init(seed, n_modes=4, amplitude=0.2))
    
    # Cosine inits
    for seed in [1, 2, 3]:
        initializations.append(_cosine_init(seed, n_harmonics=3))
    
    # Edge enhanced
    for seed in [10, 20]:
        initializations.append(_edge_enhanced_init(seed))
    
    # Random
    for seed in [30, 40, 50, 60]:
        initializations.append(_random_init(seed))
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Optimize each initialization with two-stage L-BFGS-B
    for idx, h0 in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        h0 = np.maximum(h0, 0.01)
        
        # Stage 1: Coarse optimization with higher T
        try:
            result1 = optimize.minimize(
                lambda x: soft_objective(x, T=0.03), h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 10:
                break
            
            # Stage 2: Fine optimization with lower T
            if result1.fun < best_score * 1.05:
                result2 = optimize.minimize(
                    lambda x: soft_objective(x, T=0.005), result1.x,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Evaluate both stages with exact objective
                for candidate in [result1.x, result2.x]:
                    h_new = np.maximum(candidate, 0)
                    actual_score = compute_autoconv_max(h_new)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_new.copy()
                        
        except Exception:
            pass
    
    # Final polish: projected gradient descent from best solution
    if time.time() < deadline - 5:
        try:
            # Use a very low temperature for precise optimization
            for attempt in range(5):
                if time.time() > deadline - 3:
                    break
                    
                h_perturbed = best_h + 0.003 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                
                result = optimize.minimize(
                    lambda x: soft_objective(x, T=0.002), h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_new = np.maximum(result.x, 0)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
                    
        except Exception:
            pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
