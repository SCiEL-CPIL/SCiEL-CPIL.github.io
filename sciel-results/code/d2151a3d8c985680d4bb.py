"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _value(h):
    """Compute the objective: abs(2*N*max(conv(h,h)) / sum(h)^2)"""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def construct():
    t0 = time.time()
    deadline = 145.0  # seconds, with safety margin before 150s kill
    
    # Start with seed
    best_h = np.ones(N, dtype=np.float64)
    best_val = _value(best_h)
    
    # Try various structured patterns to find good initial points
    candidates = [np.ones(N, dtype=np.float64)]
    
    # Random patterns
    np.random.seed(42)
    for _ in range(20):
        h = np.random.randn(N)
        if abs(np.sum(h)) > 1e-6:
            candidates.append(h)
    
    # Sign-flip patterns: positive in middle, negative at edges
    for frac in [0.1, 0.2, 0.3, 0.4]:
        h = np.ones(N)
        k = int(N * frac)
        h[:k] = -1.0
        h[-k:] = -1.0
        candidates.append(h)
    
    # Sine patterns
    x = np.linspace(0, 2*np.pi, N)
    for k in [1, 2, 3, 4, 5]:
        candidates.append(np.sin(k * x))
        candidates.append(np.cos(k * x))
    
    # Triangle / ramp patterns
    x = np.linspace(-1, 1, N)
    candidates.append(x)
    candidates.append(x**2)
    candidates.append(1.0 - np.abs(x))
    
    # Evaluate all candidates
    for h in candidates:
        h = h.astype(np.float64)
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Check time
    if time.time() - t0 > deadline - 60:
        return best_h
    
    # Local optimization using gradient-free methods
    # Use Nelder-Mead on a reduced parameterization
    # We'll optimize a smooth surrogate using gradient descent
    
    # Use scipy minimize with BFGS on a reparameterized problem
    # Let h = a + b * x where x is some basis, to maintain sum(h) != 0
    
    # Better approach: use the fact that the problem is scale-invariant.
    # Fix sum(h) = 1, then minimize 2*N*max(conv(h,h))
    
    # Reparameterize: h = (1/(N-1)) * (1 - e) where e_i for i=0..N-2, and e_{N-1} = -sum(e_0..e_{N-2})
    # Actually, fix sum(h)=1. Let h = 1/N + d where sum(d)=0.
    
    # Use a smooth approximation: replace max(c) with log-sum-exp
    def _smooth_max(c, tau=0.01):
        """Soft max via log-sum-exp for positive values, but c can be negative.
        Use: tau * log(sum(exp(c/tau))) as approximation to max(c)"""
        # For safety, shift to avoid overflow
        c_shifted = c - np.max(c)
        return np.max(c) + tau * np.log(np.sum(np.exp(c_shifted)))
    
    def _objective(d, tau=0.05):
        """d has N-1 elements; h = 1/N * ones + d with last component adjusted so sum(d)=0"""
        if len(d) != N - 1:
            d = d[:N-1]
        d_full = np.zeros(N)
        d_full[:N-1] = d
        d_full[N-1] = -np.sum(d)
        h = np.ones(N) / N + d_full
        # Ensure sum(h) = 1 exactly
        h = h / np.sum(h) if abs(np.sum(h)) > 1e-12 else h + 1e-10
        c = np.convolve(h, h)
        return 2.0 * N * _smooth_max(c, tau)
    
    def _objective_grad(d, tau=0.05):
        """Gradient of the smooth objective using finite differences (slow) or analytic."""
        # Use finite differences for simplicity
        eps = 1e-6
        g = np.zeros_like(d)
        f0 = _objective(d, tau)
        for i in range(len(d)):
            d_plus = d.copy()
            d_plus[i] += eps
            g[i] = (_objective(d_plus, tau) - f0) / eps
        return f0, g
    
    # Use L-BFGS-B
    d0 = np.zeros(N - 1)
    best_d = d0.copy()
    best_obj = _objective(d0)
    
    if time.time() - t0 > deadline - 45:
        # Convert best_d back to h
        d_full = np.zeros(N)
        d_full[:N-1] = best_d
        d_full[N-1] = -np.sum(best_d)
        h = np.ones(N) / N + d_full
        h = h / np.sum(h)
        return h
    
    result = minimize(
        _objective, d0, args=(0.05,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 500, 'disp': False}
    )
    
    if result.success or result.fun < best_obj:
        best_d = result.x
        best_obj = result.fun
    
    # Convert back to h
    d_full = np.zeros(N)
    d_full[:N-1] = best_d
    d_full[N-1] = -np.sum(best_d)
    h_opt = np.ones(N) / N + d_full
    h_opt = h_opt / np.sum(h_opt)
    
    v_opt = _value(h_opt)
    if v_opt < best_val:
        best_h = h_opt
        best_val = v_opt
    
    # Try from the best candidate as starting point
    if time.time() - t0 < deadline - 30:
        # Re-express best_h in the d parameterization
        h_norm = best_h / np.sum(best_h)
        d_init = h_norm - np.ones(N) / N
        d_init = d_init[:N-1]
        
        result2 = minimize(
            _objective, d_init, args=(0.03,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 300, 'disp': False}
        )
        
        if result2.fun < best_obj:
            best_d = result2.x
            best_obj = result2.fun
            d_full = np.zeros(N)
            d_full[:N-1] = best_d
            d_full[N-1] = -np.sum(best_d)
            h_opt2 = np.ones(N) / N + d_full
            h_opt2 = h_opt2 / np.sum(h_opt2)
            v_opt2 = _value(h_opt2)
            if v_opt2 < best_val:
                best_h = h_opt2
                best_val = v_opt2
    
    # Final check
    if time.time() - t0 > deadline - 5:
        pass  # Just return what we have
    
    return best_h

# EVOLVE-BLOCK-END
