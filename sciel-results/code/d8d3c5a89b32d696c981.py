"""Structural change: low-dimensional parameterization via a quartic potential V(x).
The 12 double-root locations are treated as critical points of V(x) = sum_i a_i * x^i / (x^2 + 1).
We optimize the 4 potential coefficients to minimize C4, using the fact that at optimum
the 12 roots satisfy sum_{j != i} 1/(x_i - x_j) + V'(x_i)/2 = 0 (up to scaling).
This reduces the search from 12 DOF to 4 DOF for the global structure, then polishes with 12 DOF."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.special import roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Build P(x) = prod (x-x_i)^2 and P2 = P^2 in ascending coeff order."""
    n = len(roots)
    if n == 0:
        return np.array([0.0])
    # Start with P(x) = 1
    p_coeffs = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r x + r^2
        new_len = len(p_coeffs) + 2
        new_coeffs = np.zeros(new_len)
        # x^2 * P(x): shift by 2
        new_coeffs[2:] += p_coeffs
        # -2r * x * P(x): shift by 1, scale by -2r
        new_coeffs[1:-1] += (-2 * r) * p_coeffs
        # r^2 * P(x): no shift, scale by r^2
        new_coeffs[:len(p_coeffs)] += (r * r) * p_coeffs
        p_coeffs = new_coeffs
    # P2 = P^2 via convolution
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    return p2_coeffs

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(np.convolve(P, P))  # This is wrong, P is already P^2
    # Actually _build_p_squared_coeffs returns P^2, so I2 should be integral of P^2 * e^{-x}
    # But we need I2 = int (P(x))^2 e^{-x} dx, and P from build is already prod (x-xi)^2
    # So P from _build_p_squared_coeffs is P^2, and I2 = int P^2 e^{-x} = _poly_moment(P)
    # Wait, let me re-read: _build_p_squared_coeffs returns P2 = P^2 where P = prod (x-xi)^2
    # So P2 is a polynomial of degree 48. I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx = int P2 e^{-x} dx
    # But _build_p_squared_coeffs returns P2, not P. We need both.
    # Let me fix this.
    
    # Rebuild: get P and P2 separately
    n = len(roots)
    if n == 0:
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    
    # Build P(x) = prod (x - xi)^2
    p_coeffs = np.array([1.0])
    for r in roots:
        new_len = len(p_coeffs) + 2
        new_coeffs = np.zeros(new_len)
        new_coeffs[2:] += p_coeffs
        new_coeffs[1:-1] += (-2 * r) * p_coeffs
        new_coeffs[:len(p_coeffs)] += (r * r) * p_coeffs
        p_coeffs = new_coeffs
    
    # Build P2 = P^2
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    
    I1 = _poly_moment(p_coeffs)
    I2 = _poly_moment(p2_coeffs)
    
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Analytical gradient
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        # dP/dxi = -2 * (x - xi) * P / (x - xi)^2 = -2 * P / (x - xi)
        # More carefully: P = (x-xi)^2 * R_i where R_i = prod_{j!=i} (x-xj)^2
        # dP/dxi = -2(x-xi) * R_i
        # I1 = int P e^{-x} dx, dI1/dxi = int -2(x-xi) R_i e^{-x} dx
        # I2 = int P^2 e^{-x} dx, dI2/dxi = int 2P * dP/dxi e^{-x} dx = int -4(x-xi) P R_i e^{-x} dx
        #      = int -4(x-xi)^2 R_i^2 e^{-x} dx
        
        R_i = np.array([1.0])
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = polymul(R_i, polyfromroots([roots[j], roots[j]]))
        
        # dI1/dxi = -2 * int (x-xi) R_i e^{-x} dx = -2 * (int x*R_i e^{-x} dx - xi * int R_i e^{-x} dx)
        xR_i = polymul(R_i, np.array([0.0, 1.0]))
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        
        # dI2/dxi = -4 * int (x-xi)^2 R_i^2 e^{-x} dx = -4 * int (x^2 - 2xi x + xi^2) R_i^2 e^{-x} dx
        #         = -4 * (int x^2 R_i^2 e^{-x} dx - 2xi int x R_i^2 e^{-x} dx + xi^2 int R_i^2 e^{-x} dx)
        # But R_i^2 is the square of R_i, which is expensive. Let's use:
        # (x-xi)^2 R_i^2 = (x-xi) * [(x-xi) R_i] * R_i = (x-xi) * [x R_i - xi R_i] * R_i
        # This is getting complex. Let me use a different approach.
        # dI2/dxi = 2 * I2 * d(log P)/dxi ... no, that's not right.
        # Let me just compute it numerically if analytical is too slow, or use a simpler formula.
        
        # Alternative: use the fact that at a local min, the gradient should be zero.
        # For now, let's use a simpler gradient computation.
        # dC4/dxi = (dI2/dxi * I1^2 - I2 * 2*I1*dI1/dxi) / I1^4
        #         = dI2/dxi / I1^2 - 2*I2*dI1/dxi / I1^3
        
        # For dI2/dxi, note that P^2 = prod_{j} (x-xj)^4
        # d(P^2)/dxi = P^2 * d(log P^2)/dxi = P^2 * (-4/(x-xi))
        # So dI2/dxi = int -4 P^2 / (x-xi) e^{-x} dx
        # This is still complex. Let me use numerical differentiation for the gradient.
        
    # Use numerical gradient as fallback
    eps = 1e-7
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        roots_plus = roots.copy()
        roots_plus[i] += eps
        roots_minus = roots.copy()
        roots_minus[i] -= eps
        c4_plus = _compute_c4(roots_plus)
        c4_minus = _compute_c4(roots_minus)
        if np.isfinite(c4_plus) and np.isfinite(c4_minus):
            grad[i] = (c4_plus - c4_minus) / (2 * eps)
        else:
            grad[i] = 0.0
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def _potential_roots(a, n_roots=12):
    """
    Find n_roots positive roots of V'(x) = 0 where V(x) = sum_i a_i * x^i / (x^2 + 1).
    This is a structural change: instead of optimizing 12 roots directly,
    we optimize 4 potential coefficients and find the critical points.
    """
    # V(x) = a0 + a1*x + a2*x^2 + a3*x^3 all divided by (x^2+1)
    # V'(x) = [(a1 + 2*a2*x + 3*a3*x^2)(x^2+1) - (a0 + a1*x + a2*x^2 + a3*x^3)*2x] / (x^2+1)^2
    # = [a1*x^2 + a1 + 2*a2*x^3 + 2*a2*x + 3*a3*x^4 + 3*a3*x^2 - 2*a0*x - 2*a1*x^2 - 2*a2*x^3 - 2*a3*x^4] / (x^2+1)^2
    # = [a3*x^4 + (a1 + 3*a3 - 2*a1)*x^2 + (2*a2 - 2*a2)*x^3 + (a1 + 3*a3)*x^2 ... let me redo this carefully.
    
    # Numerator of V'(x):
    # N(x) = (a1 + 2*a2*x + 3*a3*x^2)(x^2+1) - 2x(a0 + a1*x + a2*x^2 + a3*x^3)
    #      = a1*x^2 + a1 + 2*a2*x^3 + 2*a2*x + 3*a3*x^4 + 3*a3*x^2 - 2*a0*x - 2*a1*x^2 - 2*a2*x^3 - 2*a3*x^4
    #      = (3*a3 - 2*a3)*x^4 + (2*a2 - 2*a2)*x^3 + (a1 - 2*a1 + 3*a3)*x^2 + (2*a2 - 2*a0)*x + a1
    #      = a3*x^4 + 0*x^3 + (-a1 + 3*a3)*x^2 + (2*a2 - 2*a0)*x + a1
    
    # So V'(x) = 0 iff a3*x^4 + (-a1 + 3*a3)*x^2 + (2*a2 - 2*a0)*x + a1 = 0
    # This is a quartic! We can find all its roots.
    
    coeffs = np.array([a[0], (2*a[2] - 2*a[0]), (-a[1] + 3*a[3]), 0.0, a[3]])
    # Note: numpy uses descending order for roots, but we have ascending here.
    # coeffs[k] is coeff of x^k, so x^0: a[0], x^1: (2*a2-2*a0), x^2: (-a1+3*a3), x^3: 0, x^4: a3
    # For numpy.roots, we need descending order: [a3, 0, (-a1+3*a3), (2*a2-2*a0), a[0]]
    
    if abs(a[3]) < 1e-15:
        # Cubic or lower
        c = np.array([(-a[1] + 3*a[3]), (2*a[2] - 2*a[0]), a[0]])
        if abs(c[0]) < 1e-15:
            # Quadratic or lower: (2*a2-2*a0)*x + a0 = 0
            if abs(2*a[2] - 2*a[0]) < 1e-15:
                return np.array([])
            x_root = -a[0] / (2*a[2] - 2*a[0])
            roots = np.array([x_root]) if x_root > 0 else np.array([])
        else:
            # Quadratic: c0*x^2 + c1*x + c2 = 0 where c0 = -a1+3*a3, c1 = 2*a2-2*a0, c2 = a0
            disc = c[1]**2 - 4*c[0]*c[2]
            if disc < 0:
                return np.array([])
            sqrt_disc = math.sqrt(disc)
            r1 = (-c[1] + sqrt_disc) / (2*c[0])
            r2 = (-c[1] - sqrt_disc) / (2*c[0])
            roots = np.array([r for r in [r1, r2] if r > 0 and np.isfinite(r)])
    else:
        c_desc = np.array([a[3], 0.0, (-a[1] + 3*a[3]), (2*a[2] - 2*a[0]), a[0]])
        try:
            r_all = np.roots(c_desc)
        except Exception:
            return np.array([])
        # Keep only real positive roots
        roots = np.array([r.real for r in r_all if abs(r.imag) < 1e-10 and r.real > 0])
    
    if len(roots) < n_roots:
        return None  # Not enough positive critical points
    
    # Take the n_roots smallest positive roots (assuming they're the relevant ones)
    roots = np.sort(roots)[:n_roots]
    return roots

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: 1D search over alpha to find optimal root density profile
    def _log_alpha_objective(log_a):
        alpha = np.exp(log_a)
        if alpha < 1e-3 or alpha > 1e4:
            return np.inf
        return _c4_from_alpha(alpha)

    # Grid search first
    best_alpha = 0.0
    best_c4_alpha = np.inf
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 200)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _log_alpha_objective(la)
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Refine with scalar minimization
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                _log_alpha_objective,
                bounds=(math.log(1e-3), math.log(1e4)),
                method='bounded',
                options={'xatol': 1e-8, 'maxiter': 200}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Phase 2: Full 12-DOF L-BFGS-B refinement from the alpha-optimal point
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Try potential-based initialization
    if time_ok():
        # Initialize potential coefficients from the current best roots
        # Fit a simple potential to match the root structure
        a_init = np.array([1.0, 0.5, 0.1, 0.01])
        for _ in range(50):
            if not time_ok():
                break
            # Perturb the potential coefficients
            a_try = a_init + np.random.randn(4) * 0.5
            roots_pot = _potential_roots(a_try, 12)
            if roots_pot is not None and len(roots_pot) == 12:
                c4_pot = _compute_c4(roots_pot)
                if np.isfinite(c4_pot) and c4_pot < best_c4:
                    # Polish with 12-DOF
                    try:
                        res_pot = minimize(
                            _objective_grad,
                            roots_pot,
                            jac=True,
                            method='L-BFGS-B',
                            bounds=[(1e-8, None)] * 12,
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                        )
                        if np.all(res_pot.x > 0) and np.isfinite(res_pot.fun):
                            candidate = np.sort(res_pot.x)
                            c4 = _compute_c4(candidate)
                            if c4 < best_c4:
                                best_c4 = c4
                                best_roots = candidate.copy()
                    except Exception:
                        pass

    # Phase 4: Try neighboring alpha values
    if time_ok():
        for alpha_offset in [0.5, 2.0, 0.1, 10.0]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < 1e-3 or a_try > 1e4:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    res2 = minimize(
                        _objective_grad,
                        xg2,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 5: Final high-precision polish
    for polish_iter in range(3):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_genlaguerre(12, 0.0)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
