"""
Strategy: Parameterized Chirp Design
Instead of optimizing all 400 values directly, we model the sequence as a
real-valued linear chirp (Zadoff-Chu-like structure) parameterized by a
chirp rate alpha and an amplitude A. We optimize these 2 parameters to
minimize the C3' metric, leveraging the fact that chirps have excellent
auto-correlation properties (low sidelobes) by construction.
We then perform a few steps of local refinement on the resulting 400-dim vector
to break the chirp symmetry if it helps.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _chirp_sequence(alpha, A):
    """
    Generate a real-valued chirp sequence.
    x[n] = A * cos(2*pi*alpha*n^2)
    This is a standard LFM (Linear Frequency Modulation) signal.
    """
    n = np.arange(N)
    return A * np.cos(2 * np.pi * alpha * n * n)

def _chirp_metric(params):
    """
    Objective function for optimizing chirp parameters.
    params: [alpha, A]
    """
    alpha, A = params
    # alpha can be any real number. The chirp period is 1/alpha.
    # We want to avoid alpha being too close to 0 (white noise-like) or too large (rapid oscillation).
    # However, minimizing the metric will naturally find the sweet spot.
    x = _chirp_sequence(alpha, A)
    return _metric(x)

def _local_refinement(x_init, deadline, rng, max_iters=2000):
    """
    Fast local refinement using a simple gradient-free search (Nelder-Mead style 
    or random perturbations) to fine-tune the chirp output.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Use a simple random walk with adaptive step size
    step_size = 0.1
    no_improve = 0
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        if no_improve > 50:
            step_size *= 0.9
            no_improve = 0
            if step_size < 1e-4:
                break
        
        # Perturb a random subset of elements
        k = rng.integers(1, 5)
        indices = rng.choice(N, size=k, replace=False)
        
        x_trial = best_x.copy()
        for idx in indices:
            x_trial[idx] += rng.normal(0, step_size)
        
        val_trial = _metric(x_trial)
        
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # 1. Global optimization of chirp parameters (alpha, A)
    # We expect alpha to be in a range where the chirp completes several cycles over N.
    # alpha * N^2 should be large enough for "chirp-like" behavior.
    # Let's search alpha in [0.001, 0.05] and A in [0.1, 5.0].
    # Note: The metric is scale-invariant for A (numerator scales with A^2, denominator with A^2).
    # So A is effectively irrelevant for the metric value, but we keep it to allow 
    # potential numerical stability or if we later add constraints. 
    # Actually, since metric is scale invariant, we can fix A=1 and just optimize alpha.
    
    # Let's verify scale invariance:
    # x' = A*x
    # conv(x') = A^2 conv(x)
    # max|conv(x')| = A^2 max|conv(x)|
    # sum(x') = A sum(x)
    # (sum(x'))^2 = A^2 (sum(x))^2
    # metric(x') = (A^2 max|conv(x)|) / (A^2 (sum(x))^2) = metric(x)
    # Yes, scale invariant. So we fix A=1.0.
    
    best_val = float('inf')
    best_x = np.ones(N)
    
    # Optimize alpha using differential evolution (global)
    # alpha range: The phase is 2*pi*alpha*n^2.
    # For n=N-1, phase ~ 2*pi*alpha*N^2.
    # We want the instantaneous frequency to vary.
    # Let's try alpha in [0.0001, 0.01].
    
    try:
        res = differential_evolution(
            lambda a: _chirp_metric([a[0], 1.0]),
            bounds=[(0.0001, 0.01)],
            maxiter=50,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False # We will polish later
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_alpha = res.x[0]
            best_x = _chirp_sequence(best_alpha, 1.0)
            
    except Exception:
        # Fallback to a few random alphas
        for _ in range(100):
            alpha = rng.uniform(0.0001, 0.01)
            x = _chirp_sequence(alpha, 1.0)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x

    # 2. Local refinement on the 400-dim vector
    # The chirp provides a very good starting point.
    # We use the remaining time for local search.
    if time.time() < deadline - 5:
        best_x, best_val = _local_refinement(best_x, deadline - 5, rng, max_iters=5000)
        
    # 3. Final high-precision polish if time permits
    if time.time() < deadline - 2:
        # Quick coordinate descent polish
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            improved = False
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    x_trial = best_x.copy()
                    x_trial[idx] += delta
                    v = _metric(x_trial)
                    if v < best_val:
                        best_val = v
                        best_x = x_trial
                        improved = True
            if not improved:
                break

    return best_x
# EVOLVE-BLOCK-END
