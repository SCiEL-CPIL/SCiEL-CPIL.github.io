"""
Program for the third autocorrelation inequality task.

We attempt to beat the constant-vector bound (value=2.0) by using a chirp representation:
h[k] = real( exp(i * (phi0 + k * rate)) )
This introduces a linearly varying phase which may decorrelate the autocorrelation sidelobes,
potentially reducing the maximum of the autoconvolution relative to the sum squared.

We optimize the two scalar parameters (phi0, rate) using differential evolution for global search
followed by Nelder-Mead refinement.

Note: The theoretical lower bound analysis in the seed program suggested 2.0 is the minimum,
but empirical results from other solutions (scores > 1.0) indicate that the bound might be
looser than thought or that specific structures can exploit the discrete nature of the problem.
We follow the directive to try this structural change.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400

def _elapsed(t0):
    return time.time() - t0

def _chirp_heights(params):
    """
    Generate step heights from chirp parameters.
    params[0] = initial phase (phi0)
    params[1] = chirp rate (rate)
    """
    phi0, rate = params
    k = np.arange(N)
    # Real part of complex exponential: cos(phi0 + k * rate)
    h = np.cos(phi0 + k * rate)
    
    # Ensure sum is not zero. If it is, add a small constant or shift phase.
    # For a pure cosine, the sum is zero if the frequency is such that it completes
    # an integer number of half-cycles over the interval, etc.
    # To be safe, we can check and adjust.
    sum_h = np.sum(h)
    if np.isclose(sum_h, 0.0):
        # Add a small DC offset to ensure non-zero sum.
        # This changes the structure slightly but ensures validity.
        h = h + 0.1
    return h

def _objective(params):
    """
    Compute the value to be minimized.
    """
    h = _chirp_heights(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    sum_h = np.sum(h)
    if np.isclose(sum_h, 0.0):
        return 1e10
    
    # Compute autoconvolution
    c = np.convolve(h, h)
    max_c = np.max(c)
    
    value = abs(2 * N * max_c / (sum_h ** 2))
    return value

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses differential evolution to find optimal chirp parameters.
    """
    t0 = time.time()
    timeout = 140.0  # Safety margin
    
    # Define bounds for parameters
    # phi0 in [0, 2*pi]
    # rate in [0, 2*pi] (frequency variation over N steps)
    # We can also allow negative rates, but symmetry suggests [0, 2*pi] is sufficient for cos.
    bounds = [(0, 2 * np.pi), (0, 2 * np.pi)]
    
    best_value = 2.0  # Seed value
    best_params = (0.0, 0.0)  # Corresponds to h[k]=cos(0)=1, which is constant
    
    # Initial evaluation of seed
    seed_h = np.ones(N)
    seed_c = np.convolve(seed_h, seed_h)
    seed_max_c = np.max(seed_c)
    seed_sum = np.sum(seed_h)
    seed_value = abs(2 * N * seed_max_c / (seed_sum ** 2))
    if seed_value < best_value:
        best_value = seed_value
    
    # Differential Evolution
    try:
        # Use a moderate population size and max iterations to fit within time
        # N=400, convolve is O(N^2) ~ 160k ops. 1000 evals * 160k ~ 160M ops.
        # Python/NumPy can handle this, but we need to be careful.
        # Let's use maxiter=50, popsize=20 -> ~1000 evaluations.
        
        de_result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False,  # We will do our own local refinement
            init='latinhypercube'
        )
        
        if de_result.fun < best_value:
            best_value = de_result.fun
            best_params = de_result.x
            
    except Exception:
        pass
    
    # Check time
    if _elapsed(t0) > timeout:
        h = _chirp_heights(best_params)
        return h
    
    # Local refinement with Nelder-Mead
    if best_params is not None:
        try:
            nm_result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if nm_result.fun < best_value:
                best_value = nm_result.fun
                best_params = nm_result.x
                
        except Exception:
            pass
    
    # Check time again
    if _elapsed(t0) > timeout:
        h = _chirp_heights(best_params)
        return h
        
    # Generate final heights
    h = _chirp_heights(best_params)
    
    # Final safety checks
    if not np.all(np.isfinite(h)):
        h = np.ones(N)
    
    sum_h = np.sum(h)
    if np.isclose(sum_h, 0.0):
        h = np.ones(N)
        
    return h

# EVOLVE-BLOCK-START
# No additional logic needed. The construct function handles the optimization.
# EVOLVE-BLOCK-END
