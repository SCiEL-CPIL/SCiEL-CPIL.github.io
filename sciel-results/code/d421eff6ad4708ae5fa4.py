"""Second autocorrelation inequality optimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 50
# Equal intervals on [-1/4, 1/4]
x = np.linspace(-0.25, 0.25, N)
dx = x[1] - x[0]

# Precompute indices for convolution
# autoconvolution: c_k = sum_i h_i * h_{i+k} (discrete)
# With N=50 points, convolution has 2N-1 = 99 points
# The "supremum norm of the autoconvolution" is max |c_k|
# The squared L2 norm: sum h_i^2 (times dx for continuous, but ratio is scale-invariant in dx)
# L1 norm: sum h_i
# We want to maximize: (sum h_i^2) / (sum h_i * max_k |c_k|)
# where c_k = sum_i h_i * h_{i+k}

def _score(h):
    """Compute the ratio (L2^2) / (L1 * sup|autoconv|) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return -np.inf
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    # autoconvolution
    c = fftconvolve(h, h, mode='full')
    sup_c = np.max(np.abs(c))
    if sup_c < 1e-15:
        return -np.inf
    return l2_sq / (l1 * sup_c)

def _objective_neg(h):
    """Negative of score for minimization."""
    s = _score(h)
    if np.isinf(s) or s < 0:
        return 1e10
    return -s

def _dft_parameterize(z):
    """Parameterize nonnegative heights via |FFT|^2 of complex z."""
    # z is complex vector of length N
    H = np.abs(np.fft.fft(z)) ** 2
    # Scale to have reasonable magnitude
    return H / (N * N)

def _from_dft_params(params):
    """Convert real parameter vector to nonnegative heights via DFT."""
    # params: real vector of length 2N (real and imaginary parts)
    z = params[:N] + 1j * params[N:]
    H = np.abs(np.fft.fft(z)) ** 2
    return H

def _to_dft_params(h):
    """Convert nonnegative heights to DFT parameter vector."""
    # We need to find z such that |FFT(z)|^2 = h
    # This is a phase retrieval problem; for simplicity, use sqrt(h) as magnitude
    # and random phases
    mag = np.sqrt(h)
    # Use random phases
    phases = 2 * np.pi * np.random.rand(N)
    z = mag * np.exp(1j * phases)
    return np.concatenate([z.real, z.imag])

def _smooth_objective(params, temperature=0.01):
    """Smoothed objective that handles the supremum term softly."""
    h = _from_dft_params(params)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    c = fftconvolve(h, h, mode='full')
    # Soft max of |c|
    abs_c = np.abs(c)
    # Use log-sum-exp for smooth max
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 1e10
    # Soft max: sum exp(c_i / T) * T
    soft_max = temperature * np.log(np.sum(np.exp(abs_c / temperature)))
    # Actually, use a simpler soft max
    # soft_max = np.max(abs_c) + temperature * np.log(np.mean(np.exp((abs_c - max_c)/temperature)))
    soft_max = max_c + temperature * np.log(np.mean(np.exp((abs_c - max_c) / temperature)))
    score = l2_sq / (l1 * soft_max)
    return -score

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Strategy 1: DFT parameterization with gradient-based optimization
    # Multiple diverse initializations
    
    def _time_remaining():
        return deadline - time.time()
    
    # Diverse initial shapes for DFT
    inits = []
    
    # 1. Uniform (current seed)
    inits.append(np.ones(N))
    
    # 2. Gaussian centered
    idx = np.arange(N)
    center = N // 2
    sigma = N / 4
    inits.append(np.exp(-0.5 * ((idx - center) / sigma) ** 2))
    
    # 3. Lorentzian
    inits.append(1.0 / (1.0 + ((idx - center) / (N / 6)) ** 2))
    
    # 4. Power law (decay from center)
    inits.append(1.0 / (1.0 + np.abs(idx - center) ** 1.5))
    
    # 5. Two peaks
    inits.append(np.exp(-0.5 * ((idx - N//4) / (N/8)) ** 2) + np.exp(-0.5 * ((idx - 3*N//4) / (N/8)) ** 2))
    
    # 6. Cosine window
    inits.append(0.5 * (1 + np.cos(2 * np.pi * idx / N)))
    
    # 7. Triangular
    inits.append(1.0 - 2.0 * np.abs(idx - center) / N)
    
    # 8. Sinc-like
    inits.append(np.sinc((idx - center) / (N/3)))
    inits[-1] = np.maximum(inits[-1], 0)
    
    # For each initialization, run DFT-based optimization
    for init_h in inits:
        if _time_remaining() < 5.0:
            break
        
        # Convert to DFT parameters
        mag = np.sqrt(init_h)
        # Try a few random phase patterns
        for phase_seed in range(3):
            if _time_remaining() < 5.0:
                break
            
            rng = np.random.RandomState(phase_seed * 100 + 7)
            phases = 2 * np.pi * rng.rand(N)
            z = mag * np.exp(1j * phases)
            params0 = np.concatenate([z.real, z.imag])
            
            # Gradient-based optimization on DFT parameters
            try:
                result = minimize(
                    _smooth_objective,
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8},
                    args=(0.01,)
                )
                
                if result.success or result.fun < -best_score:
                    h_candidate = _from_dft_params(result.x)
                    s = _score(h_candidate)
                    if s > best_score:
                        best_score = s
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Strategy 2: Direct optimization with differential evolution on nonnegative h
    if _time_remaining() > 15.0:
        bounds = [(0, 1)] * N
        try:
            de_result = differential_evolution(
                _objective_neg,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True
            )
            h_candidate = np.maximum(de_result.x, 0)
            s = _score(h_candidate)
            if s > best_score:
                best_score = s
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Strategy 3: Local refinement with Nelder-Mead on the best found
    if _time_remaining() > 10.0:
        try:
            nm_result = minimize(
                _objective_neg,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            h_candidate = np.maximum(nm_result.x, 0)
            s = _score(h_candidate)
            if s > best_score:
                best_score = s
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final: ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
