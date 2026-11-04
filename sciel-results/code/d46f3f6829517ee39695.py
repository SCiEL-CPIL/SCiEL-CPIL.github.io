"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400

# EVOLVE-BLOCK-START

def _autoconv_peak(x):
    """Compute max absolute autoconvolution divided by squared integral."""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.inf
    fft_size = 2 * N
    X = np.fft.fft(x, n=fft_size)
    auto = np.fft.ifft(X * X).real
    max_abs = np.max(np.abs(auto))
    return max_abs / (integral * integral)


def _spectral_to_signal(params, n_harmonics):
    """Convert spectral parameters to a time-domain signal.
    
    params layout: [amplitude_0, phase_0, amplitude_1, phase_1, ..., amplitude_{K-1}, phase_{K-1}]
    Signal: x[n] = sum_k amp_k * cos(2*pi*k*n/N + phase_k)
    """
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(n_harmonics):
        amp = params[2 * k]
        phase = params[2 * k + 1]
        x += amp * np.cos(2 * np.pi * k * n / N + phase)
    return x


def _spectral_objective(params, n_harmonics):
    """Objective for differential evolution in spectral domain."""
    x = _spectral_to_signal(params, n_harmonics)
    return _autoconv_peak(x)


def _local_polish(x0, deadline, max_passes=30):
    """Local sign-flipping polish on a continuous signal.
    
    First binarize to +/-1, then do local search.
    """
    # Binarize: sign of each element (with small threshold)
    x = np.sign(x0)
    x[np.abs(x) < 1e-10] = 1.0  # Handle zeros
    current_score = _autoconv_peak(x)
    
    for p in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        idx = np.random.permutation(N)
        for j in idx:
            if time.time() > deadline:
                break
            old_val = x[j]
            x[j] = -old_val
            new_score = _autoconv_peak(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x[j] = old_val
        if not improved:
            break
    
    # Also try continuous local moves: adjust individual values slightly
    for p in range(10):
        if time.time() > deadline:
            break
        improved = False
        idx = np.random.permutation(N)
        for j in idx:
            if time.time() > deadline:
                break
            # Try small continuous perturbation
            delta = 0.1 * np.random.choice([-1, 1])
            old_val = x[j]
            x[j] = old_val + delta
            new_score = _autoconv_peak(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                x[j] = old_val
        if not improved:
            break
    
    return x, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_x = np.ones(N)
    best_score = _autoconv_peak(best_x)
    
    n_harmonics = 12  # Use 12 harmonics (24 parameters)
    
    # Phase 1: Global optimization in spectral domain using differential evolution
    # This reduces dimensionality from 400 to 2*n_harmonics = 24
    bounds = [(0.0, 1.0), (0.0, 2 * np.pi)] * n_harmonics
    
    # Multiple DE runs with different random seeds for diversity
    n_de_runs = 5
    de_time_budget = 60  # seconds for DE phase
    
    for run_i in range(n_de_runs):
        if time.time() > deadline - de_time_budget / n_de_runs:
            break
        
        seed = 42 + run_i * 1000
        try:
            result = differential_evolution(
                _spectral_objective,
                bounds,
                args=(n_harmonics,),
                seed=seed,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            
            if result.fun < best_score:
                x_candidate = _spectral_to_signal(result.x, n_harmonics)
                # Local polish
                x_polished, score_polished = _local_polish(x_candidate, deadline)
                if score_polished < best_score:
                    best_score = score_polished
                    best_x = x_polished.copy()
        except Exception:
            continue
    
    # Phase 2: Additional local search from diverse starting points
    if time.time() < deadline - 10:
        n = np.arange(N)
        additional_starts = [
            np.random.choice([-1, 1], N),
            np.cos(np.pi * n * (n + 1) / N),  # Zadoff-Chu like
            np.cos(np.pi * n**2 / N),  # Chirp
            np.sin(2 * np.pi * n / N),  # Single sine
            np.ones(N),
        ]
        
        for x0 in additional_starts:
            if time.time() > deadline - 5:
                break
            x_polished, score_polished = _local_polish(x0, deadline, max_passes=20)
            if score_polished < best_score:
                best_score = score_polished
                best_x = x_polished.copy()
    
    return best_x

# EVOLVE-BLOCK-END
