"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 180.0  # seconds; keep safety margin


def _hermite_probabilists(n, x):
    """Compute probabilists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0 = np.ones_like(x)
    h1 = x.copy()
    for k in range(2, n + 1):
        h2 = x * h1 - (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_objective(params):
    """
    Compute the uncertainty product A(g) * A(F(g)) for the Hermite-Gaussian test function.
    params = [a0, a4, a8]; a12 is derived from g(0)=0.
    """
    a0, a4, a8 = params
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 945.0

    # Grid
    L = 30.0  # half-width of integration domain
    N = 8192  # number of grid points
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    # Build g(x)
    h0 = _hermite_probabilists(0, x)
    h4 = _hermite_probabilists(4, x)
    h8 = _hermite_probabilists(8, x)
    h12 = _hermite_probabilists(12, x)

    g = (a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12) * np.exp(-x**2 / 2.0)

    # A(g) = integral |g(x)| dx
    A_g = np.sum(np.abs(g)) * dx

    # Fourier transform: use the property that F[H_n(x) e^{-x^2/2}] = i^n H_n(xi) e^{-xi^2/2}
    # But for the L1 norm of the FT, we need |F(g)(xi)| integrated.
    # Since g is a linear combination, F(g) = a0*F(H0 e^{-x^2/2}) + a4*F(H4 e^{-x^2/2}) + ...
    # F(H_n(x) e^{-x^2/2})(xi) = i^n H_n(xi) e^{-xi^2/2} (with the convention F(f)(xi) = int f(x) e^{-2pi i x xi} dx)
    # Actually, let's be careful. The standard result: if f(x) = H_n(x) e^{-x^2/2}, then
    # F(f)(xi) = (2pi)^{1/2} i^n H_n(xi) e^{-xi^2/2} under the unitary convention, or
    # with F(f)(xi) = int f(x) e^{-i x xi} dx, we get F(f)(xi) = sqrt(2pi) i^n H_n(xi) e^{-xi^2/2}
    
    # Let's use the convention F(f)(xi) = int_{-inf}^{inf} f(x) e^{-i x xi} dx
    # Then F(H_n(x) e^{-x^2/2})(xi) = sqrt(2*pi) * i^n * H_n(xi) * e^{-xi^2/2}
    
    # For the uncertainty product, the constant sqrt(2*pi) factors out and we can work with
    # the "shape" of the FT. The actual A(F(g)) = sqrt(2*pi) * integral |a0*H0(xi) + a4*i^4*H4(xi) + a8*i^8*H8(xi) + a12*i^12*H12(xi)| e^{-xi^2/2} dxi
    # i^4 = 1, i^8 = 1, i^12 = -1
    # So F(g)(xi) = sqrt(2*pi) * (a0*H0(xi) + a4*H4(xi) + a8*H8(xi) - a12*H12(xi)) * e^{-xi^2/2}
    
    xi = x  # symmetric grid
    h0_xi = _hermite_probabilists(0, xi)
    h4_xi = _hermite_probabilists(4, xi)
    h8_xi = _hermite_probabilists(8, xi)
    h12_xi = _hermite_probabilists(12, xi)
    
    # i^0=1, i^4=1, i^8=1, i^12=-1
    g_hat_shape = (a0 * h0_xi + a4 * h4_xi + a8 * h8_xi + (-a12) * h12_xi) * np.exp(-xi**2 / 2.0)
    
    A_Fg = np.sum(np.abs(g_hat_shape)) * dx * np.sqrt(2 * np.pi)
    
    if A_g < 1e-15 or A_Fg < 1e-15:
        return 1e10  # avoid division by zero / degenerate case
    
    return A_g * A_Fg


def _objective_wrapper(params):
    """Wrapper for scipy minimize."""
    try:
        val = _compute_objective(params)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """
    Return three coefficients [a0, a4, a8] of the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    The score is the upper bound on C4; lower is better.
    """
    # Seed values
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_objective(seed)
    best_params = seed.copy()

    deadline = _start_time + _DEADLINE - 10.0  # 10s safety margin

    # Try multiple optimization strategies
    try:
        # L-BFGS-B from seed
        res = minimize(_objective_wrapper, seed, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 500})
        if time.time() > deadline:
            break
        if res.success or res.fun < best_val:
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline:
        return best_params

    # Try Nelder-Mead from seed
    try:
        res2 = minimize(_objective_wrapper, seed, method='Nelder-Mead',
                        options={'maxiter': 500, 'maxfev': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if time.time() > deadline:
            break
        if np.isfinite(res2.fun) and res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass

    if time.time() > deadline:
        return best_params

    # Try a few random restarts
    rng = np.random.RandomState(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        perturbation = rng.normal(0, 0.5, size=3)
        x0 = seed + perturbation
        try:
            res3 = minimize(_objective_wrapper, x0, method='L-BFGS-B',
                            options={'maxiter': 100, 'maxfun': 300})
            if np.isfinite(res3.fun) and res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass

    if time.time() > deadline:
        return best_params

    # Polished local search around best
    try:
        res4 = minimize(_objective_wrapper, best_params, method='Powell',
                        options={'maxiter': 200, 'maxfev': 1000, 'xtol': 1e-8, 'ftol': 1e-10})
        if np.isfinite(res4.fun) and res4.fun < best_val:
            best_val = res4.fun
            best_params = res4.x.copy()
    except Exception:
        pass

    return best_params
# EVOLVE-BLOCK-END
