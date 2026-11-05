"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses a structured parameterization (DC + Sine modes) to reduce dimensionality,
    followed by Differential Evolution for global search and L-BFGS-B for local refinement.
    This approach aims to flatten the power spectrum to minimize the autocorrelation peak.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0

    def compute_value(h):
        """Compute the evaluation metric efficiently."""
        if not np.all(np.isfinite(h)):
            return np.inf
        h_sum = np.sum(h)
        if abs(h_sum) < 1e-10:
            return np.inf
        
        # Normalize to sum=1 to simplify calculation, then scale back?
        # The formula is | 2*N*max_conv / sum^2 |.
        # If we scale h by c, max_conv scales by c^2, sum^2 scales by c^2.
        # So the value is invariant to scaling.
        # We can just compute with the raw h.
        
        # Use FFT for linear convolution
        # Zero-pad to 2N to avoid circular convolution wrap-around issues for lags 0..N-1
        n_fft = 2 * N
        H = fft(h, n=n_fft)
        # Power spectrum
        P = H * H
        # Inverse FFT to get autocorrelation
        # The linear autocorrelation r[k] for k=0..N-1 is in the first N elements
        # of the circular convolution of the zero-padded signal.
        autoconv_circ = ifft(P, n=n_fft).real
        # The first N elements correspond to lags -(N-1) to 0? 
        # Actually, for real signals, r[k] = r[-k].
        # np.convolve(h, h) has length 2N-1.
        # Index N-1 is lag 0.
        # The circular conv of length 2N has indices 0..2N-1.
        # autoconv_circ[0:N] corresponds to the first N values of the linear conv.
        # Due to symmetry, max(linear_conv) == max(autoconv_circ[0:N])
        max_conv = np.max(autoconv_circ[:N])
        
        value = abs(2 * N * max_conv / (h_sum ** 2))
        return value

    # Precompute sine basis
    # We use M modes
    M = 30
    j = np.arange(N)
    # Sine modes: sin(2*pi*m*j/N)
    # These have zero mean.
    sine_basis = np.zeros((M, N))
    for m in range(1, M + 1):
        sine_basis[m-1] = np.sin(2 * np.pi * m * j / N)

    def model_h(params):
        """Reconstruct h from parameters: [A, B_1, ..., B_M]"""
        A = params[0]
        Bs = params[1:]
        h = np.full(N, A)
        h += sine_basis @ Bs
        return h

    def objective_params(params):
        """Objective function for parameter optimization."""
        h = model_h(params)
        return compute_value(h)

    # Strategy:
    # 1. Run Differential Evolution on the low-dimensional parameter space.
    # 2. Take the best result and refine with L-BFGS-B in the full space.
    
    # Bounds for DE
    # A should be positive to keep sum away from 0.
    # B_m can be negative.
    bounds = [(0.5, 5.0)] + [(-1.0, 1.0)] * M
    
    np.random.seed(42)
    
    # Run DE with limited budget
    # popsize=10, maxiter=50. 
    # 10*50 = 500 evaluations.
    # Each eval is fast (FFT).
    try:
        result_de = differential_evolution(
            objective_params,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            tol=1e-5,
            polish=False, # We do our own polish
            callback=lambda x, convergence: None
        )
        if result_de.fun < best_value:
            best_params = result_de.x
            best_h_candidate = model_h(best_params)
            best_value = result_de.fun
            best_h = best_h_candidate.copy()
    except Exception:
        pass

    # Local refinement with L-BFGS-B on the full h vector
    # Initialize from the best DE result
    if best_value < 2.0:
        h_init = best_h.copy()
    else:
        # Fallback: random initialization
        h_init = np.ones(N)
        h_init += 0.1 * np.random.randn(N)
        
    def objective_full(h):
        return compute_value(h)

    try:
        result_lbfgs = minimize(
            objective_full,
            h_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result_lbfgs.fun < best_value:
            best_value = result_lbfgs.fun
            best_h = result_lbfgs.x.copy()
    except Exception:
        pass

    # If we have time, try a second DE run with different seed and more modes?
    # Or just a second L-BFGS-B from a different random start?
    if time.time() < deadline - 10.0:
        # Try a few random restarts of L-BFGS-B
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
            h_rand = np.ones(N)
            h_rand += 0.5 * np.random.randn(N)
            try:
                res = minimize(
                    objective_full,
                    h_rand,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_h = res.x.copy()
            except Exception:
                pass

    return best_h
# EVOLVE-BLOCK-END
