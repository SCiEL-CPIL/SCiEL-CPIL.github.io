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
    """Smooth soft-max objective with fixed scale (sum=1) for L-BFGS-B."""
    h = np.maximum(x, 0)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10
    # Normalize to fix scale, reducing effective dimensionality
    h = h / total
    conv = _fft_convolve(h)
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -100, 100)
    soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
    return soft_max

def _flat_spectrum_init(seed, n_modes=6, amplitude=0.25):
    """Generate initialization targeting flat power spectrum via random phases."""
    rng = np.random.RandomState(seed)
    
    # Random phases for flat spectrum
    phases = rng.uniform(0, 2 * np.pi, N)
    
    # Ensure real signal: H[N-k] = conj(H[k])
    # For unit magnitude: phase[N-k] = -phase[k]
    for k in range(1, N // 2):
        phases[N - k] = -phases[k]
    phases[N // 2] = 0.0  # Nyquist frequency
    
    # Add some structured low-frequency content
    if n_modes > 0:
        freqs = rng.choice(range(1, N // 2), size=min(n_modes, N // 2 - 1), replace=False)
        for f in freqs:
            phase = rng.uniform(0, 2 * np.pi)
            phases[f] = phase
            phases[N - f] = -phase
    
    # Complex spectrum with unit magnitude
    H = np.exp(1j * phases)
    
    # Inverse FFT
    h = np.real(ifft(H, N))
    
    # Make nonnegative
    h = np.abs(h)
    if np.max(h) > 0:
        h = h / np.max(h)
    
    # Mix with uniform for stability
    h = (1 - amplitude) * np.ones(N) + amplitude * h
    h = np.maximum(h, 0.005)
    h = h / np.sum(h)
    
    return h

def _gaussian_envelope_init(seed, n_peaks=2, width_range=(0.1, 0.4)):
    """Initialize with Gaussian-like envelope to reduce central concentration."""
    rng = np.random.RandomState(seed)
    t = np.linspace(-1, 1, N)
    
    h = np.ones(N) * 0.3
    for _ in range(n_peaks):
        center = rng.uniform(-0.8, 0.8)
        width = rng.uniform(*width_range)
        h += np.exp(-0.5 * ((t - center) / width) ** 2)
    
    h = np.maximum(h, 0.005)
    h = h / np.sum(h)
    return h

def _cosine_init(seed, n_harmonics=4):
    """Generate initialization using cosine harmonics."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, N, endpoint=False)
    
    h = np.ones(N)
    for k in range(1, n_harmonics + 1):
        freq = rng.uniform(1, 10)
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.08 / k
        h += amp * np.cos(2 * np.pi * freq * t + phase)
    
    h = np.maximum(h, 0.005)
    h = h / np.sum(h)
    return h

def _edge_enhanced_init(seed):
    """Enhance edges to reduce central peak."""
    rng = np.random.RandomState(seed)
    t = np.linspace(-1, 1, N)
    h = 1.0 + 0.4 * np.abs(t) + 0.03 * rng.randn(N)
    h = np.maximum(h, 0.005)
    h = h / np.sum(h)
    return h

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    h = rng.rand(N) + 0.05
    h = h / np.sum(h)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin for 150s limit
    
    # Initial guess: all ones
    h = np.ones(N)
    best_score = compute_autoconv_max(h)
    best_h = h.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # Flat spectrum inits
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        initializations.append(_flat_spectrum_init(seed, n_modes=5, amplitude=0.25))
    
    # Gaussian envelope inits
    for seed in [1, 2, 3, 4]:
        initializations.append(_gaussian_envelope_init(seed, n_peaks=2, width_range=(0.15, 0.45)))
    
    # Cosine inits
    for seed in [10, 11, 12]:
        initializations.append(_cosine_init(seed, n_harmonics=4))
    
    # Edge enhanced
    for seed in [20, 21, 22]:
        initializations.append(_edge_enhanced_init(seed))
    
    # Random
    for seed in [30, 31, 32, 33]:
        initializations.append(_random_init(seed))
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Optimize each initialization with multi-stage L-BFGS-B
    for idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        h0 = np.maximum(h0, 0.005)
        h0 = h0 / np.sum(h0)
        
        # Stage 1: Coarse optimization with higher T
        try:
            result1 = optimize.minimize(
                lambda x: soft_objective(x, T=0.05), h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 15:
                break
            
            # Stage 2: Medium optimization
            if result1.fun < best_score * 1.1:
                result2 = optimize.minimize(
                    lambda x: soft_objective(x, T=0.01), result1.x,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Stage 3: Fine optimization
                if time.time() < deadline - 10 and result2.fun < best_score * 1.05:
                    result3 = optimize.minimize(
                        lambda x: soft_objective(x, T=0.003), result2.x,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    
                    # Evaluate all stages with exact objective
                    for candidate in [result1.x, result2.x, result3.x]:
                        h_new = np.maximum(candidate, 0)
                        actual_score = compute_autoconv_max(h_new)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_h = h_new.copy()
                else:
                    for candidate in [result1.x, result2.x]:
                        h_new = np.maximum(candidate, 0)
                        actual_score = compute_autoconv_max(h_new)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_h = h_new.copy()
                        
        except Exception:
            pass
    
    # Final polish: multiple perturbations from best solution
    if time.time() < deadline - 5:
        try:
            for attempt in range(8):
                if time.time() > deadline - 3:
                    break
                    
                # Small random perturbation
                h_perturbed = best_h + 0.002 * np.random.randn(N)
                h_perturbed = np.maximum(h_perturbed, 0)
                h_perturbed = h_perturbed / np.sum(h_perturbed)
                
                result = optimize.minimize(
                    lambda x: soft_objective(x, T=0.001), h_perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_new = np.maximum(result.x, 0)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
                    
        except Exception:
            pass
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 2:
        try:
            h_perturbed = best_h.copy()
            h_perturbed = h_perturbed / np.sum(h_perturbed)
            
            result = optimize.minimize(
                lambda x: soft_objective(x, T=0.0005), h_perturbed,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13}
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
