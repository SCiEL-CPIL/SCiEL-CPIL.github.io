"""
Optimized program for the suite task.
Strategy: Use a low-rank spline parameterization (rank-2/3) with differential evolution
to allow non-smooth, non-Gaussian shapes that can better approximate the theoretical
flat-top window. The step function is represented as a linear combination of a few
basis functions (low-rank), reducing dimensionality while allowing flexible shapes.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600

def _compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    if len(heights) == 0:
        return np.inf
    h = np.asarray(heights, dtype=np.float64)
    s = h.sum()
    if s <= 0:
        return np.inf
    # Autoconvolution via FFT
    n = len(h)
    fft_size = 2 * n
    H = np.fft.rfft(h, n=fft_size)
    auto = np.fft.irfft(H * np.conj(H), n=fft_size)[:n]
    max_val = np.max(auto)
    return max_val / (s * s)

def _build_basis_functions(n_basis, N):
    """Build a set of basis functions for low-rank representation.
    
    We use a combination of:
    - A constant (flat) basis
    - Cosine modes (smooth, periodic-like)
    - Linear/ramp basis
    - Gaussian-like bump at center
    This allows representing flat-top windows and other non-Gaussian shapes.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 / N
    
    basis = []
    
    # Constant basis
    basis.append(np.ones(N))
    
    # Cosine modes (capture oscillatory behavior)
    for k in range(1, min(n_basis - 2, 8) + 1):
        basis.append(np.cos(2 * np.pi * k * x))
        basis.append(np.sin(2 * np.pi * k * x))
    
    # Add a central Gaussian bump
    width = 0.1
    basis.append(np.exp(-0.5 * (x / width) ** 2))
    
    # Add a flat-top window approximation (raised cosine)
    cutoff = 0.2
    flat_top = np.where(np.abs(x) < cutoff, 1.0, 0.0)
    # Smooth the edges
    edge_width = 0.05
    for i in range(N):
        xi = x[i]
        if cutoff - edge_width < abs(xi) < cutoff + edge_width:
            t = (abs(xi) - (cutoff - edge_width)) / (2 * edge_width)
            flat_top[i] = 0.5 * (1 + np.cos(np.pi * t))
    basis.append(flat_top)
    
    # Trim to requested number of basis functions
    basis = basis[:n_basis]
    
    return np.array(basis)  # shape: (n_basis, N)

def _param_to_heights(params, basis, n_coeff):
    """Convert low-rank parameters to step heights.
    
    params: array of coefficients
    basis: array of basis functions, shape (n_basis, N)
    n_coeff: number of coefficients to use
    """
    coeffs = params[:n_coeff]
    heights = basis[:n_coeff] @ coeffs
    
    # Ensure nonnegativity
    heights = np.maximum(heights, 0.0)
    
    # Normalize
    s = heights.sum()
    if s > 0:
        heights /= s
    else:
        heights = np.ones(N) / N
    
    return heights

def _objective_low_rank(params, basis, n_coeff):
    """Objective function for low-rank optimization."""
    heights = _param_to_heights(params, basis, n_coeff)
    return _compute_score(heights)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with uniform as baseline
    uniform_heights = np.ones(N)
    best_heights = uniform_heights.copy()
    best_score = _compute_score(best_heights)
    
    # Try different numbers of basis functions (low-rank)
    for n_basis in [4, 6, 8, 10, 12]:
        if time.time() > deadline - 25:
            break
        
        # Build basis functions
        basis = _build_basis_functions(n_basis, N)
        n_coeff = n_basis
        
        # Define bounds for coefficients
        # Coefficients can be positive or negative, but we clamp to nonnegative heights
        # Use reasonable bounds
        bounds = [(-2.0, 2.0)] * n_coeff
        
        # Differential evolution
        np.random.seed(42 + n_basis)
        
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            try:
                result = differential_evolution(
                    _objective_low_rank,
                    bounds,
                    args=(basis, n_coeff),
                    seed=42 + n_basis + restart,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    polish=True
                )
                
                if result.fun < best_score:
                    heights = _param_to_heights(result.x, basis, n_coeff)
                    score = _compute_score(heights)
                    if score < best_score:
                        best_score = score
                        best_heights = heights.copy()
            except Exception:
                pass
    
    # Return best heights found
    return list(best_heights)

# EVOLVE-BLOCK-END
