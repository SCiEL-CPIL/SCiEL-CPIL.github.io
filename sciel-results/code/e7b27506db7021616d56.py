"""Optimized program: root-based parameterization with direct optimization of q's roots."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 200.0

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

# Precompute Hermite polynomial coefficients (physicists' convention)
def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]
        prev2 = H[k - 2]
        curr = [0.0] * (k + 1)
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) for given coefficients."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for coeff, H in [(c0, _H0), (c1, _H4), (c2, _H8)]:
        for i in range(len(H)):
            g_coeffs[i] += coeff * H[i]
    
    for i in range(len(_H12)):
        g_coeffs[i] += c3 * _H12[i]
    
    lead_coeff = g_coeffs[12]
    if lead_coeff < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    if abs(g_coeffs[0]) > 1e-15 or abs(g_coeffs[1]) > 1e-15:
        return float('inf')
    
    q_coeffs = g_coeffs[2:]
    
    q0 = q_coeffs[0]
    q2 = q_coeffs[2]
    q4 = q_coeffs[4]
    q6 = q_coeffs[6]
    q8 = q_coeffs[8]
    q10 = q_coeffs[10]
    
    z_coeffs = np.array([q10, q8, q6, q4, q2, q0])
    
    if np.allclose(z_coeffs, 0, atol=1e-30):
        return np.inf
    
    roots = np.roots(z_coeffs)
    
    valid_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-10 and rt.real > 1e-15:
            z_val = rt.real
            eps = 1e-8 * max(1.0, abs(z_val))
            Q_minus = q10*(z_val-eps)**5 + q8*(z_val-eps)**4 + q6*(z_val-eps)**3 + q4*(z_val-eps)**2 + q2*(z_val-eps) + q0
            Q_plus = q10*(z_val+eps)**5 + q8*(z_val+eps)**4 + q6*(z_val+eps)**3 + q4*(z_val+eps)**2 + q2*(z_val+eps) + q0
            if Q_minus * Q_plus < 0:
                valid_roots.append(z_val)
    
    if not valid_roots:
        return np.inf
    
    r_max = np.sqrt(max(valid_roots))
    return r_max**2 / (2 * np.pi)

# Root-based parameterization
# q(x) = a * prod_{i=1}^{5} (x^2 - z_i) where z_i > 0
# The value is max(z_i) / (2*pi)
# We want to minimize max(z_i) subject to the constraint that the coefficients
# correspond to some valid (c0, c1, c2, c3) with g(0)=0 and leading coeff positive.

# The monomial coefficients of q are:
# q(x) = a * (x^10 - (sum z_i) x^8 + (sum z_i z_j) x^6 - ... + (-1)^5 prod z_i)
# So q10 = a, q8 = -a*sum(z), q6 = a*sum(z_i z_j), etc.

# We need to find (c0, c1, c2, c3) such that:
# g(x) = x^2 * q(x) = x^2 * a * prod(x^2 - z_i)
# and g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12

# This gives a linear system: for each monomial coefficient of g, we have
# g_coeff[m] = c0*H0[m] + c1*H4[m] + c2*H8[m] + c3*H12[m]

# g has degree 12, so coefficients from x^0 to x^12.
# g(x) = x^2 * q(x), so g[0]=0, g[1]=0, g[2]=q0, g[4]=q2, ..., g[12]=q10=a

# We have 4 unknowns (c0,c1,c2,c3) and 13 equations (g[0]..g[12]).
# But g[0]=0 and g[1]=0 are automatically satisfied.
# Also c3 is determined by g(0)=0: c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0)/H12_0

# So effectively 3 free parameters (c0,c1,c2) and c3 is determined.
# The constraint is that the resulting g must match x^2*q(x) for some q with positive roots.

# Alternative approach: directly optimize over z_1,...,z_5 > 0 and a > 0,
# then solve for (c0,c1,c2,c3) via least squares, and check validity.

def _q_to_g_coeffs(z_roots, a):
    """Given roots z_i and leading coeff a, compute g coefficients."""
    # q(x) = a * prod(x^2 - z_i)
    # Build q coefficients in ascending power of x^2
    # Let y = x^2, Q(y) = a * prod(y - z_i)
    z = np.array(z_roots)
    # Polynomial with roots z_i, leading coeff a
    # Q(y) = a * (y^5 - e1*y^4 + e2*y^3 - e3*y^2 + e4*y - e5)
    # where e_k are elementary symmetric polynomials
    
    # Compute elementary symmetric polynomials
    e1 = np.sum(z)
    e2 = sum(z[i]*z[j] for i in range(5) for j in range(i+1, 5))
    e3 = sum(z[i]*z[j]*z[k] for i in range(5) for j in range(i+1,5) for k in range(j+1,5))
    e4 = sum(z[i]*z[j]*z[k]*z[l] for i in range(5) for j in range(i+1,5) for k in range(j+1,5) for l in range(k+1,5))
    e5 = np.prod(z)
    
    # Q(y) = a*(y^5 - e1*y^4 + e2*y^3 - e3*y^2 + e4*y - e5)
    # q(x) = Q(x^2) = a*x^10 - a*e1*x^8 + a*e2*x^6 - a*e3*x^4 + a*e4*x^2 - a*e5
    
    # g(x) = x^2 * q(x) = a*x^12 - a*e1*x^10 + a*e2*x^8 - a*e3*x^6 + a*e4*x^4 - a*e5*x^2
    
    g_coeffs = [0.0] * 13
    g_coeffs[12] = a
    g_coeffs[10] = -a * e1
    g_coeffs[8] = a * e2
    g_coeffs[6] = -a * e3
    g_coeffs[4] = a * e4
    g_coeffs[2] = -a * e5
    
    return g_coeffs

def _solve_for_hermite_coeffs(g_coeffs):
    """Solve for (c0, c1, c2, c3) given g coefficients."""
    # g[m] = c0*H0[m] + c1*H4[m] + c2*H8[m] + c3*H12[m]
    # Build matrix A where A[i,j] = H_j[i] for j=0,1,2,3 corresponding to H0,H4,H8,H12
    # We use rows 2,4,6,8,10,12 (even powers >= 2) since g[0]=g[1]=0
    
    rows = [2, 4, 6, 8, 10, 12]
    A = np.zeros((len(rows), 4))
    b = np.zeros(len(rows))
    
    for idx, row in enumerate(rows):
        A[idx, 0] = _H0[row] if row < len(_H0) else 0.0
        A[idx, 1] = _H4[row] if row < len(_H4) else 0.0
        A[idx, 2] = _H8[row] if row < len(_H8) else 0.0
        A[idx, 3] = _H12[row] if row < len(_H12) else 0.0
        b[idx] = g_coeffs[row]
    
    # Solve least squares
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
    
    c0, c1, c2, c3 = coeffs
    
    # Verify g(0) = 0
    g0_check = c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0 + c3 * _H12_0
    if abs(g0_check) > 1e-10:
        return None
    
    # Verify leading coefficient is positive
    if g_coeffs[12] < 0:
        return None
    
    return (c0, c1, c2)

def objective_root_based(params):
    """Objective: minimize max(z_i) / (2*pi) where z_i are the roots."""
    # params: [z1, z2, z3, z4, z5, log_a]
    z_roots = params[:5]
    log_a = params[5]
    
    if np.any(z_roots <= 0):
        return 1e10
    
    a = np.exp(log_a)
    if a <= 0:
        return 1e10
    
    g_coeffs = _q_to_g_coeffs(z_roots, a)
    
    hermite_coeffs = _solve_for_hermite_coeffs(g_coeffs)
    if hermite_coeffs is None:
        return 1e10
    
    c0, c1, c2 = hermite_coeffs
    
    # Verify by computing the actual value
    val = compute_value(c0, c1, c2)
    if np.isinf(val) or np.isnan(val):
        return 1e10
    
    return val

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = compute_value(seed[0], seed[1], seed[2])
    best_params = seed.copy()
    
    # Phase 1: Optimize in root space
    if _time_left() > 120:
        # Initial guess: spread out roots
        rng = np.random.RandomState(42)
        
        # Try multiple initial configurations
        best_root_val = 1e10
        best_root_params = None
        
        for trial in range(30):
            # Random positive roots
            z_init = rng.uniform(0.1, 5.0, 5)
            # Sort them
            z_init = np.sort(z_init)
            log_a_init = np.log(rng.uniform(0.1, 10.0))
            
            params_init = np.concatenate([z_init, [log_a_init]])
            
            try:
                res = minimize(objective_root_based, params_init, method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 2000, 'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_root_val:
                    best_root_val = res.fun
                    best_root_params = res.x.copy()
            except:
                continue
            
            if _time_left() < 60:
                break
        
        if best_root_params is not None and best_root_val < best_val:
            # Convert back to hermite coefficients
            z_roots = best_root_params[:5]
            a = np.exp(best_root_params[5])
            g_coeffs = _q_to_g_coeffs(z_roots, a)
            hermite_coeffs = _solve_for_hermite_coeffs(g_coeffs)
            if hermite_coeffs is not None:
                c0, c1, c2 = hermite_coeffs
                val = compute_value(c0, c1, c2)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_params = np.array([c0, c1, c2])
    
    # Phase 2: Differential evolution in coefficient space
    if _time_left() > 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-2, 2)]
            de_result = differential_evolution(
                lambda p: compute_value(*p) if np.isfinite(compute_value(*p)) else 1e10,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if np.isfinite(de_result.fun) and de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish
    if _time_left() > 30:
        try:
            res = minimize(lambda p: compute_value(*p) if np.isfinite(compute_value(*p)) else 1e10,
                          best_params, method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'maxfev': 5000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = compute_value(best_params[0], best_params[1], best_params[2])
    if np.isinf(final_val) or np.isnan(final_val) or final_val > best_val * 1.01:
        return seed
    
    return best_params
# EVOLVE-BLOCK-END
