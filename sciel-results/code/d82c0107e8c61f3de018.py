"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 195.0  # 15s safety margin before 210s kill

# Precompute the analytic integrals needed for the bound.
# g(x) = exp(-x^2/2) * P(x) where P = c0*H0 + c4*H4 + c8*H8 + c12*H12
# c12 = -(c0 + 12*c4 + 1344*c8) / 467712
#
# ||g||_2^2 = integral exp(-x^2) * P(x)^2 dx
# ||g||_1 = integral exp(-x^2/2) * |P(x)| dx
#
# For ||g||_2^2: expand P^2 = sum_{i,j} c_i c_j H_i H_j
# integral exp(-x^2) H_i(x) H_j(x) dx is known analytically.
# For physicist's Hermite: integral exp(-x^2) H_n(x) H_m(x) dx = sqrt(pi) * 2^n * n! * delta_{nm}
# Wait, that's for the standard normalization. Let me verify:
# The physicist's Hermite polynomials satisfy:
# integral_{-inf}^{inf} exp(-x^2) H_n(x) H_m(x) dx = sqrt(pi) * 2^n * n! * delta_{nm}
#
# So the cross terms vanish! This means ||g||_2^2 = sqrt(pi) * sum c_i^2 * 2^{d_i} * d_i!
# where d_i is the degree of each Hermite polynomial.
#
# Degrees: H0 (deg 0), H4 (deg 4), H8 (deg 8), H12 (deg 12)
# integral exp(-x^2) H_0^2 dx = sqrt(pi) * 1 = sqrt(pi)
# integral exp(-x^2) H_4^2 dx = sqrt(pi) * 2^4 * 4! = sqrt(pi) * 16 * 24 = sqrt(pi) * 384
# integral exp(-x^2) H_8^2 dx = sqrt(pi) * 2^8 * 8! = sqrt(pi) * 256 * 40320 = sqrt(pi) * 10321920
# integral exp(-x^2) H_12^2 dx = sqrt(pi) * 2^12 * 12! = sqrt(pi) * 4096 * 479001600 = sqrt(pi) * 1961987635200
#
# For ||g||_1: integral exp(-x^2/2) |P(x)| dx - this is harder due to the absolute value.
# We need to find where P(x) changes sign and integrate piecewise.

def _hermite_vals(x):
    """Compute H0, H4, H8, H12 at points x (array)."""
    x = np.asarray(x, dtype=float)
    H0 = np.ones_like(x)
    H4 = 16.0*x**4 - 48.0*x**2 + 12.0
    H8 = 256.0*x**8 - 4608.0*x**6 + 23040.0*x**4 - 34560.0*x**2 + 1344.0
    H12 = 4096.0*x**12 - 184320.0*x**10 + 3225600.0*x**8 - 22809600.0*x**6 + 7077880.0*x**4 - 84672000.0*x**2 + 467712.0
    return H0, H4, H8, H12

# Precompute the L2 norm squared coefficients
# ||g||_2^2 = sqrt(pi) * (c0^2 * 1 + c4^2 * 384 + c8^2 * 10321920 + c12^2 * 1961987635200)
_L2_COEFFS = np.array([1.0, 384.0, 10321920.0, 1961987635200.0])
_L2_SCALE = np.sqrt(np.pi)

def _P_from_coeffs(c0, c4, c8, x):
    """Compute P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12 at points x."""
    c12 = -(c0 + 12.0*c4 + 1344.0*c8) / 467712.0
    H0, H4, H8, H12 = _hermite_vals(x)
    return c0*H0 + c4*H4 + c8*H8 + c12*H12

def _compute_L1(c0, c4, c8):
    """Compute ||g||_1 = integral exp(-x^2/2) |P(x)| dx using numerical integration."""
    # We need to handle the absolute value. Use quad with splitting at roots.
    import scipy.integrate
    
    def integrand(x):
        P = _P_from_coeffs(c0, c4, c8, np.array([x]))[0]
        return np.exp(-x**2 / 2.0) * abs(P)
    
    # Integrate over a range where the Gaussian is negligible
    # Use adaptive quadrature
    result, _ = scipy.integrate.quad(integrand, -15, 15, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result

def _compute_L2_sq(c0, c4, c8):
    """Compute ||g||_2^2 analytically."""
    c12 = -(c0 + 12.0*c4 + 1344.0*c8) / 467712.0
    coeffs = np.array([c0, c4, c8, c12])
    return _L2_SCALE * np.sum(_L2_COEFFS * coeffs**2)

def _compute_bound(c0, c4, c8):
    """Compute the upper bound J = (||g||_1)^2 / ||g||_2^2."""
    L2_sq = _compute_L2_sq(c0, c4, c8)
    if L2_sq < 1e-15:
        return 1e10
    L1 = _compute_L1(c0, c4, c8)
    return (L1 * L1) / L2_sq

def _compute_bound_grad(c0, c4, c8, eps=1e-7):
    """Compute bound and gradient using central finite differences."""
    f0 = _compute_bound(c0, c4, c8)
    grad = np.zeros(3)
    for i in range(3):
        h = eps * max(1.0, abs([c0, c4, c8][i]))
        params_plus = np.array([c0, c4, c8])
        params_minus = np.array([c0, c4, c8])
        params_plus[i] += h
        params_minus[i] -= h
        fp = _compute_bound(*params_plus)
        fm = _compute_bound(*params_minus)
        grad[i] = (fp - fm) / (2.0 * h)
    return f0, grad

def _remez_exchange(start_coeffs, max_rounds=20, tol=1e-10):
    """
    Remez exchange for the 3-parameter uncertainty bound.
    At each round, identify the direction of steepest ascent of the
    bound's local landscape (via gradient) and perform a 1-D minimization
    along that direction, then re-evaluate. This approximates equioscillation
    of the error surface by iteratively pushing the worst-case direction.
    """
    params = np.array(start_coeffs, dtype=float)
    best_f = _compute_bound(*params)
    best_params = params.copy()
    
    for r in range(max_rounds):
        if time.time() - _START_TIME > _DEADLINE - 15:
            break
        f, grad = _compute_bound_grad(params[0], params[1], params[2])
        if f < best_f:
            best_f = f
            best_params = params.copy()
        
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Direction of steepest descent
        direction = -grad / gnorm
        
        # 1-D minimization along this direction using golden section
        a = -1.0
        b = 1.0
        # Expand interval to bracket a minimum
        f_center = _compute_bound(*(params + 0.5 * direction))
        if f_center < best_f:
            # Try expanding
            for scale in [1.0, 2.0, 4.0, 8.0]:
                f_test = _compute_bound(*(params + scale * direction))
                if f_test < best_f:
                    a = 0.0
                    b = scale * 2.0
                    break
                else:
                    b = scale
                    a = scale / 2.0
                    break
        else:
            # Minimum is behind us; try negative direction
            direction = -direction
            for scale in [1.0, 2.0, 4.0, 8.0]:
                f_test = _compute_bound(*(params + scale * direction))
                if f_test < best_f:
                    a = 0.0
                    b = scale * 2.0
                    break
                else:
                    b = scale
                    a = scale / 2.0
                    break
        
        # Golden section search in [a, b]
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _compute_bound(*(params + c * direction))
        fd = _compute_bound(*(params + d * direction))
        
        for _ in range(40):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _compute_bound(*(params + c * direction))
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _compute_bound(*(params + d * direction))
        
        # Best point from 1-D search
        t_best = (a + b) / 2.0
        f_best_1d = _compute_bound(*(params + t_best * direction))
        
        if f_best_1d < best_f:
            best_f = f_best_1d
            best_params = params + t_best * direction
        
        # Update params to the new best
        params = best_params.copy()
        
        # Convergence check
        if abs(f - best_f) < 1e-12:
            break
    
    return best_params, best_f

def construct():
    """Optimize the three Hermite coefficients to minimize the uncertainty bound."""
    # Seed values
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed_coeffs.copy()
    best_score = _compute_bound(*seed_coeffs)
    
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    def objective_grad(params):
        f, g = _compute_bound_grad(params[0], params[1], params[2])
        return f, g
    
    # Primary optimization from seed
    try:
        res = scipy.optimize.minimize(
            objective,
            seed_coeffs,
            method='L-BFGS-B',
            jac=objective_grad,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Multi-start with random points
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 30:
        np.random.seed(42)
        n_starts = 30
        for i in range(n_starts):
            if time.time() - _START_TIME > _DEADLINE - 15:
                break
            # Generate random starting point
            scale = 1.0
            random_coeffs = np.array([
                np.random.uniform(-3.0, 3.0) * scale,
                np.random.uniform(-0.5, 0.5) * scale,
                np.random.uniform(-0.05, 0.05) * scale
            ])
            try:
                res_r = scipy.optimize.minimize(
                    objective,
                    random_coeffs,
                    method='L-BFGS-B',
                    jac=objective_grad,
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                if res_r.fun < best_score:
                    best_score = res_r.fun
                    best_coeffs = res_r.x.copy()
            except Exception:
                pass
    
    # Final polish from best found
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 10:
        try:
            res_final = scipy.optimize.minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                jac=objective_grad,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass
    
    # Remez exchange polish: iterate steepest-direction 1-D minimization
    # to equiosculate the local error surface
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 8:
        try:
            remez_coeffs, remez_score = _remez_exchange(best_coeffs, max_rounds=30, tol=1e-12)
            if remez_score < best_score:
                best_score = remez_score
                best_coeffs = remez_coeffs.copy()
        except Exception:
            pass
    
    # One more L-BFGS-B pass from Remez result for final convergence
    current_time = time.time()
    if current_time - _START_TIME < _DEADLINE - 5:
        try:
            res_polish = scipy.optimize.minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                jac=objective_grad,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
            )
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_coeffs = res_polish.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
