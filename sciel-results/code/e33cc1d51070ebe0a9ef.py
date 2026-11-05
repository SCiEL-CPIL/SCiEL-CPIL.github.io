"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with seed
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Strategy: Use differential evolution on a low-dimensional parameterization
    # We'll represent h as a sum of a few cosine modes with varying frequencies and phases.
    # h[i] = a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N + phi_k)
    # This keeps the signal smooth and band-limited, which tends to minimize the autocorrelation peak.
    
    K = 10  # Number of cosine modes
    
    def _decode(params):
        """Decode parameters into h array."""
        a0 = params[0]
        h = np.full(N, a0)
        for k in range(1, K + 1):
            a_k = params[1 + (k-1)*2]
            phi_k = params[2 + (k-1)*2]
            h += a_k * np.cos(2 * np.pi * k * np.arange(N) / N + phi_k)
        return h
    
    def _objective(params):
        h = _decode(params)
        return _value(h)
    
    # Initial guess: mostly flat with small perturbations
    x0 = np.zeros(2 * K + 1)
    x0[0] = 1.0  # DC component
    
    # Bounds for each parameter
    # DC component: [0.5, 2.0] to keep sum positive and reasonable
    # Amplitudes: [-1.0, 1.0]
    # Phases: [0, 2*pi]
    bounds = [(0.5, 2.0)]
    for k in range(K):
        bounds.append((-1.0, 1.0))  # amplitude
        bounds.append((0, 2 * np.pi))  # phase
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            polish=True,
            init='sobol',
            popsize=15
        )
        
        if result.fun < best_val:
            best_h = _decode(result.x)
            best_val = result.fun
    except Exception:
        pass
    
    # If time remains, try a second round with different initialization
    if time.time() < deadline - 20:
        try:
            # Try with random initial amplitudes
            x0_rand = np.random.randn(2 * K + 1) * 0.1
            x0_rand[0] = 1.0
            
            # Use a different seed and more iterations
            result2 = differential_evolution(
                _objective,
                bounds,
                seed=123,
                maxiter=30,
                tol=1e-8,
                polish=True,
                init='latinhypercube',
                popsize=20
            )
            
            if result2.fun < best_val:
                best_h = _decode(result2.x)
                best_val = result2.fun
        except Exception:
            pass
    
    # Final check: ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
