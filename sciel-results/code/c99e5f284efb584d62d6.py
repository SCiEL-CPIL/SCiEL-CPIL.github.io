"""Spectral parameterization for signed autocorrelation minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = fftconvolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _construct_signal_from_params(params):
    """
    Construct a binary signal from spectral parameters.
    params: array of shape (3*M,) -> [A1, f1, p1, A2, f2, p2, ...]
    """
    M = len(params) // 3
    if M == 0:
        return np.ones(N)
    
    t = np.arange(N)
    signal = np.zeros(N)
    
    for k in range(M):
        A = params[3*k]
        f = params[3*k + 1]
        p = params[3*k + 2]
        # Ensure frequency is positive and within reasonable bounds
        # f is in cycles/sample. We want low to mid frequencies.
        # Map f from [0, 1] to [0.01, 0.5] cycles/sample?
        # DE will optimize in the defined bounds.
        signal += A * np.sin(2 * np.pi * f * t + p)
        
    # Threshold to binary +/- 1
    # If signal is 0, assign 1 (arbitrary)
    x = np.where(signal >= 0, 1.0, -1.0)
    return x

def _objective(params):
    """Objective function for differential evolution."""
    x = _construct_signal_from_params(params)
    return _compute_score(x)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Number of sinusoidal components
    # More components = higher fidelity to a flat spectrum, but higher dimensionality.
    # Let's try M=10 (30 dims) and M=20 (60 dims).
    # DE is stochastic, so we can run multiple trials or increase popsize.
    
    best_score = float('inf')
    best_seq = None
    
    # Bounds for parameters:
    # Amplitude: [0, 1] (normalized)
    # Frequency: [0.001, 0.5] (cycles/sample, up to Nyquist)
    # Phase: [0, 2*pi]
    
    # We will try a few different M values or just pick a good one.
    # Based on "reduces dimensionality from 400 to ~31", M=10 (30 params) seems aligned.
    
    M = 10
    bounds = []
    for _ in range(M):
        bounds.append((0.0, 1.0))  # Amplitude
        bounds.append((0.001, 0.5)) # Frequency
        bounds.append((0.0, 2 * np.pi)) # Phase
        
    # Initial population size and max iterations
    # DE converges reasonably fast.
    # We have ~140s.
    # Let's use a moderate popsize and maxiter.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='latinhypercube'
        )
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_seq = _construct_signal_from_params(result.x)
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find anything or timed out, fall back to a simple heuristic
    if best_seq is None:
        # Random start
        rng = np.random.RandomState(42)
        best_seq = rng.choice([-1, 1], N).astype(float)
        best_score = _compute_score(best_seq)
        
    # Local polish: Single flip hill climbing
    if time.time() < deadline:
        current = best_seq.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Randomize order to avoid bias, or just iterate
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline - 5:
                    break
                current[i] *= -1
                new_score = _compute_score(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_seq = current.copy()
                    improved = True
                else:
                    current[i] *= -1 # Revert
                    
    return best_seq
# EVOLVE-BLOCK-END
