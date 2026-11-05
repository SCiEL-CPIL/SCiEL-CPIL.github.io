"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the raw value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _build_chebyshev_basis(num_coeffs):
    """Build Chebyshev basis matrix on [-1/4, 1/4] mapped to [-1, 1]."""
    x = np.linspace(-1, 1, N)
    basis = np.zeros((N, num_coeffs))
    for k in range(num_coeffs):
        basis[:, k] = np.cos(k * np.arccos(x))
    return basis

def _coeffs_to_h(coeffs, basis):
    """Convert coefficient vector to step heights."""
    return coeffs @ basis.T

def _objective_from_coeffs(coeffs, basis):
    """Objective function in coefficient space."""
    h = _coeffs_to_h(coeffs, basis)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    """Compute value and gradient analytically for full-dim optimization."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    m = idx
    dh_conv = np.zeros(n)
    valid_j = np.arange(n)
    km = m - valid_j
    mask = (km >= 0) & (km < n)
    dh_conv[mask] = 2.0 * h[km[mask]]
    
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    inits.append(np.ones(N))
    inits.append(np.linspace(1, -1, N))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    inits.append(np.exp(-100 * x**2))
    inits.append(np.exp(-10 * x**2))
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    inits.append(1.0 - 2.0 * np.abs(x))
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    inits.append(np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2))
    inits.append(np.sign(np.sin(3*np.pi*x)))
    inits.append(2.0 * (x - np.floor(x + 0.5)))
    inits.append(1.0 + 0.3 * np.sin(8 * np.pi * x))
    inits.append(1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x))
    inits.append(1.0 + 0.3 * rng.standard_normal(N))
    
    from scipy.ndimage import gaussian_filter1d
    noise = rng.standard_normal(N)
    inits.append(1.0 + 0.5 * gaussian_filter1d(noise, sigma=5))
    inits.append(np.linspace(-1, 2, N))
    inits.append(np.concatenate([np.linspace(-1, 1, N//2), np.linspace(1, -1, N - N//2)]))
    inits.append(1.0 - 0.5 * x**2)
    
    return inits

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Low-dimensional Chebyshev basis search using differential evolution
    # This reduces the search space from 400 to ~8-10 dimensions
    num_coeffs_list = [6, 8, 10]
    basis_cache = {}
    
    for num_coeffs in num_coeffs_list:
        if not _check_time():
            break
        
        if num_coeffs not in basis_cache:
            basis_cache[num_coeffs] = _build_chebyshev_basis(num_coeffs)
        basis = basis_cache[num_coeffs]
        
        # Define bounds for differential evolution
        # c0 should be positive to keep sum(h) > 0
        bounds = [(0.5, 5.0)] + [(-2.0, 2.0)] * (num_coeffs - 1)
        
        try:
            def de_objective(coeffs):
                h = _coeffs_to_h(coeffs, basis)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    return 1e10
                conv = _autoconv(h)
                return abs(2.0 * N * np.max(conv) / (s * s))
            
            result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = _coeffs_to_h(result.x, basis).copy()
                
        except Exception:
            continue
    
    # Phase 2: Try each initial condition with moderate gradient-based optimization
    inits = _make_initial_conditions()
    
    for init_h in inits:
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxfun': 600,
                    'maxls': 30,
                }
            )
            
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 3: Polish the best result
    if best_val < 2.0 and _check_time():
        try:
            result = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 800,
                    'maxls': 40,
                }
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: If time remains, try random perturbations around best
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(123)
        for _ in range(5):
            if not _check_time():
                break
            perturbation = 0.05 * rng.standard_normal(N)
            h_try = best_h + perturbation
            try:
                result = minimize(
                    _objective_and_grad,
                    h_try,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-13,
                        'gtol': 1e-11,
                        'maxfun': 400,
                        'maxls': 25,
                    }
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
            except Exception:
                continue
    
    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
