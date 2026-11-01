import numpy as np
from math import lgamma, pi, sqrt
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190.0  # 20s safety margin before 210s kill

# Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13
def _fact_ratio(n):
    if n < 0:
        return 0.0
    if n == 0:
        return sqrt(pi)
    log_val = lgamma(2*n + 1) - n * np.log(4.0) - lgamma(n + 1) + 0.5 * np.log(pi)
    return np.exp(log_val)

_I = np.array([_fact_ratio(n) for n in range(14)])

# Hermite polynomial coefficients (physicist's convention)
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395

# Polynomial coefficients for P(x) = sum_{k=0}^{6} p_k x^{2k}
# where P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
# and a12 = -(a0 + 3*a4 + 105*a8) / 10395

def _build_p(a0, a4, a8):
    """Build the polynomial coefficient array p[0..6] for P(x)."""
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 10395.0
    
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 10395.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 302400.0*a12
    p[2] = a4 + 210.0*a8 + 166320.0*a12
    p[3] = -28.0*a8 - 30240.0*a12
    p[4] = a8 + 2520.0*a12
    p[5] = -84.0*a12
    p[6] = a12
    return p

def _moments(p, q):
    """Compute moments m_k and n_k for position and Fourier sides."""
    m = np.zeros(3)
    n = np.zeros(3)
    for k in range(3):
        for i in range(7):
            pi_val = p[i]
            if pi_val == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    m[k] += pi_val * p[j] * _I[idx]
                    n[k] += 2.0*pi * q[i] * q[j] * _I[idx]
    return m, n

def _compute_bound(params):
    """Compute the uncertainty product bound."""
    a0, a4, a8 = params
    p = _build_p(a0, a4, a8)
    
    # Fourier side: Q(xi) = P(-i*xi), so q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    m, n = _moments(p, q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18
    
    return sqrt(vf * vF)

def _bound_and_grad(params):
    """Compute bound and analytic gradient using finite differences with adaptive step."""
    bound = _compute_bound(params)
    if not np.isfinite(bound) or bound > 1e10:
        return bound, np.array([1.0, 1.0, 1.0])
    
    grad = np.zeros(3)
    for i in range(3):
        # Adaptive step size based on parameter magnitude
        scale = max(1.0, abs(params[i]))
        eps = 1e-8 * scale
        
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        
        f_plus = _compute_bound(params_plus)
        f_minus = _compute_bound(params_minus)
        
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2.0 * eps)
        else:
            grad[i] = 0.0
    
    return bound, grad

def _reparameterize_to_sphere(params):
    """Reparameterize to unit sphere coordinates for stochastic search."""
    norm = np.linalg.norm(params)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return params / norm

def _sphere_to_params(sphere_coords, radius):
    """Convert sphere coordinates back to parameter space."""
    return sphere_coords * radius

def _stochastic_sphere_search(initial_params, max_iters=5000, initial_step=0.1):
    """Stochastic local search on the unit sphere with adaptive step sizes."""
    best_params = initial_params.copy()
    best_val = _compute_bound(best_params)
    
    # Normalize to sphere
    best_sphere = _reparameterize_to_sphere(best_params)
    best_radius = np.linalg.norm(best_params)
    
    rng = np.random.default_rng(12345)
    step_size = initial_step
    no_improve_count = 0
    patience = 200
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        # Generate random perturbation on the sphere
        delta = rng.normal(0, step_size, 3)
        new_sphere = best_sphere + delta
        new_sphere_norm = np.linalg.norm(new_sphere)
        if new_sphere_norm > 1e-15:
            new_sphere = new_sphere / new_sphere_norm
        
        # Also try varying the radius slightly
        radius_scale = 1.0 + rng.normal(0, 0.05)
        new_radius = best_radius * radius_scale
        
        new_params = _sphere_to_params(new_sphere, new_radius)
        new_val = _compute_bound(new_params)
        
        if np.isfinite(new_val) and new_val < best_val:
            best_val = new_val
            best_params = new_params.copy()
            best_sphere = new_sphere.copy()
            best_radius = new_radius
            no_improve_count = 0
            step_size = min(step_size * 1.05, 0.5)
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                step_size *= 0.9
                no_improve_count = 0
        
        # Occasionally restart from a random point
        if it % 500 == 0 and it > 0:
            rand_sphere = rng.normal(0, 1, 3)
            rand_sphere = rand_sphere / np.linalg.norm(rand_sphere)
            rand_radius = rng.uniform(0.1, 3.0)
            rand_params = _sphere_to_params(rand_sphere, rand_radius)
            rand_val = _compute_bound(rand_params)
            if rand_val < best_val:
                best_val = rand_val
                best_params = rand_params.copy()
                best_sphere = rand_sphere.copy()
                best_radius = rand_radius
                step_size = initial_step
    
    return best_params, best_val

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = _compute_bound(seed)
    
    best_val = seed_val
    best_params = seed.copy()
    
    # Phase 1: Gradient-based optimization from multiple starting points
    from scipy.optimize import minimize
    
    starts = [
        seed,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([0.5, 0.05, 0.0005]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.06, 0.0006]),
        np.array([-2.5, 0.12, 0.0012]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.normal(0, 1.0, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        
        try:
            res = minimize(_bound_and_grad, start, method='L-BFGS-B', jac=True,
                          options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < np.inf and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Stochastic sphere search from the best point found
    if time.time() < _DEADLINE - 20:
        sphere_params, sphere_val = _stochastic_sphere_search(best_params, max_iters=8000, initial_step=0.08)
        if sphere_val < best_val:
            best_val = sphere_val
            best_params = sphere_params
    
    # Phase 3: Final gradient polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_bound_and_grad, best_params, method='L-BFGS-B', jac=True,
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < np.inf and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more stochastic refinement
    if time.time() < _DEADLINE - 5:
        sphere_params, sphere_val = _stochastic_sphere_search(best_params, max_iters=5000, initial_step=0.03)
        if sphere_val < best_val:
            best_val = sphere_val
            best_params = sphere_params
    
    return np.array(best_params)
