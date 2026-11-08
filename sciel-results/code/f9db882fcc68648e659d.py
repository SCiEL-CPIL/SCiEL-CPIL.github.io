import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomials (physicists' convention)
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
# H12 = 4096x^12 - 98304x^10 + 846720x^8 - 3932160x^6 + 8064000x^4 - 6048000x^2 + 1039500

def _hermite_polynomial_coeffs(degree):
    """Return coefficients of physicists' Hermite polynomial H_degree as [c0, c2, c4, ..., c_degree] for even degrees."""
    if degree == 0:
        return np.array([1.0])
    elif degree == 2:
        return np.array([-2.0, 4.0])
    elif degree == 4:
        return np.array([12.0, -48.0, 16.0])
    elif degree == 8:
        return np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
    elif degree == 12:
        return np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])
    else:
        raise ValueError(f"Unsupported degree {degree}")

# Precompute Hermite polynomial coefficients
H0_coeffs = _hermite_polynomial_coeffs(0)
H4_coeffs = _hermite_polynomial_coeffs(4)
H8_coeffs = _hermite_polynomial_coeffs(8)
H12_coeffs = _hermite_polynomial_coeffs(12)

def _compute_P_coeffs(c0, c1, c2, c3):
    """Compute the monomial coefficients of P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    P = np.zeros(7)
    P[0] += c0 * H0_coeffs[0]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    return P

def _integral_x2n_exp_minus_x2(n):
    """∫_{-∞}^{∞} x^(2n) * exp(-x²) dx = (2n-1)!! * sqrt(pi) / 2^n"""
    if n == 0:
        return np.sqrt(np.pi)
    result = 1.0
    for k in range(n):
        result *= (2*k + 1)
    return result * np.sqrt(np.pi) / (2.0 ** n)

def _integral_x2n_exp_minus_2x2(n):
    """∫_{-∞}^{∞} x^(2n) * exp(-2x²) dx = (2n-1)!! * sqrt(pi) / 2^(2n+0.5)"""
    if n == 0:
        return np.sqrt(np.pi / 2.0)
    result = 1.0
    for k in range(n):
        result *= (2*k + 1)
    return result * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))

def _poly_square_moment(P, integral_func):
    """Compute ∫ P(x)² * exp(-x²) dx or similar, where P(x) = sum P[i]*x^(2i)."""
    n = len(P)
    result = 0.0
    for i in range(n):
        for j in range(n):
            result += P[i] * P[j] * integral_func(i + j)
    return result

def _poly_fourth_moment(P, integral_func):
    """Compute ∫ P(x)⁴ * exp(-2x²) dx, where P(x) = sum P[i]*x^(2i)."""
    n = len(P)
    result = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    result += P[i] * P[j] * P[k] * P[l] * integral_func(i + j + k + l)
    return result

def _objective_and_grad(c0, c1, c2):
    """Compute the objective (upper bound on C4) and its gradient."""
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    L2_sq = _poly_square_moment(P, _integral_x2n_exp_minus_x2)
    L4_4th = _poly_fourth_moment(P, _integral_x2n_exp_minus_2x2)
    
    if L2_sq <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    dP_dc0 = _compute_P_coeffs(1, 0, 0, 0) + (-1/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    dP_dc1 = _compute_P_coeffs(0, 1, 0, 0) + (-12/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    dP_dc2 = _compute_P_coeffs(0, 0, 1, 0) + (-3360/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    
    def _dL2_sq(P, dP):
        n = len(P)
        result = 0.0
        for i in range(n):
            for j in range(n):
                result += 2.0 * dP[i] * P[j] * _integral_x2n_exp_minus_x2(i + j)
        return result
    
    def _dL4_4th(P, dP):
        n = len(P)
        result = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        result += 4.0 * dP[i] * P[j] * P[k] * P[l] * _integral_x2n_exp_minus_2x2(i + j + k + l)
        return result
    
    dL2_sq_dc0 = _dL2_sq(P, dP_dc0)
    dL2_sq_dc1 = _dL2_sq(P, dP_dc1)
    dL2_sq_dc2 = _dL2_sq(P, dP_dc2)
    
    dL4_4th_dc0 = _dL4_4th(P, dP_dc0)
    dL4_4th_dc1 = _dL4_4th(P, dP_dc1)
    dL4_4th_dc2 = _dL4_4th(P, dP_dc2)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_dc0 = (0.5 * dL4_4th_dc0 / sqrt_L4 - dL2_sq_dc0 / L2_sq) * C4
    dC4_dc1 = (0.5 * dL4_4th_dc1 / sqrt_L4 - dL2_sq_dc1 / L2_sq) * C4
    dC4_dc2 = (0.5 * dL4_4th_dc2 / sqrt_L4 - dL2_sq_dc2 / L2_sq) * C4
    
    return C4, np.array([dC4_dc0, dC4_dc1, dC4_dc2])

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    # Multi-start BFGS optimization with tighter tolerances and more iterations
    best_C4 = np.inf
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Generate multiple starting points
    rng = np.random.RandomState(42)
    starts = [best_c.copy()]
    
    # Add perturbed starts around the known good point with finer granularity
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        for _ in range(5):
            perturbation = rng.normal(0, scale, 3)
            starts.append(best_c + perturbation)
    
    # Add some structured starts based on symmetry considerations
    for sign in [-1, 1]:
        for c0_scale in [0.5, 1.0, 2.0]:
            for c1_scale in [0.5, 1.0, 2.0]:
                for c2_scale in [0.5, 1.0, 2.0]:
                    candidate = np.array([
                        sign * c0_scale * best_c[0],
                        sign * c1_scale * best_c[1],
                        sign * c2_scale * best_c[2]
                    ])
                    starts.append(candidate)
    
    # Also add some random starts in a wider range
    for _ in range(30):
        starts.append(rng.uniform(-1000, 1000, 3))
    
    # Sort by initial objective to try promising ones first
    start_objs = []
    for s in starts:
        try:
            obj, _ = _objective_and_grad(s[0], s[1], s[2])
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    for obj, s in start_objs:
        if time.time() > deadline - 10.0:
            break
        if obj >= best_C4:
            continue  # Skip if initial value is already worse
        
        try:
            # First pass: BFGS with moderate tolerance
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                s,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-12}
            )
            
            current_best = result.fun if result.success else obj
            current_c = result.x if result.success else s
            
            # Second pass: High-precision polish if we have time
            if time.time() < deadline - 5.0 and np.isfinite(current_best):
                result2 = minimize(
                    lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                    current_c,
                    jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                    method='BFGS',
                    options={'maxiter': 1000, 'gtol': 1e-15}
                )
                if result2.success and result2.fun < current_best:
                    current_best = result2.fun
                    current_c = result2.x
            
            if np.isfinite(current_best) and current_best < best_C4:
                best_C4 = current_best
                best_c = current_c
                
        except Exception:
            pass
    
    # Final high-precision polish on the best solution found
    if time.time() < deadline - 2.0 and np.isfinite(best_C4):
        try:
            result_final = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 2000, 'gtol': 1e-16}
            )
            if result_final.success and result_final.fun < best_C4:
                best_C4 = result_final.fun
                best_c = result_final.x
        except Exception:
            pass
    
    return np.array(best_c)
