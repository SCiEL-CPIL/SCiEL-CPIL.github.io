"""Uncertainty inequality optimization with exact moment integrals and L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 200.0  # 10s safety margin before 210s kill

# Precompute Gaussian moments: I[n] = integral x^(2n) exp(-x^2) dx
_N_MOMENTS = 30
_I = np.zeros(_N_MOMENTS)
_I[0] = np.sqrt(np.pi)
for _n in range(1, _N_MOMENTS):
    _I[_n] = (2.0 * _n - 1.0) / 2.0 * _I[_n - 1]

# Hermite polynomial coefficients: H_n(x) = sum_k h[n,k] x^k
# Using physicist's convention: H_0=1, H_1=2x, H_{n}=2xH_{n-1}-2(n-1)H_{n-2}
def _hermite_coeffs(n):
    """Return list of coefficients [c0, c1, ..., cn] for H_n(x) = sum c_k x^k."""
    if n == 0:
        return [1.0]
    elif n == 1:
        return [0.0, 2.0]
    else:
        h_nm2 = [1.0] + [0.0] * (n - 1)  # H_0
        h_nm1 = [0.0, 2.0] + [0.0] * (n - 1)  # H_1
        for k in range(2, n + 1):
            h_n = [0.0] * (k + 1)
            # 2x * h_nm1
            for j in range(len(h_nm1)):
                if j + 1 <= k:
                    h_n[j + 1] += 2.0 * h_nm1[j]
            # -2(k-1) * h_nm2
            for j in range(len(h_nm2)):
                h_n[j] -= 2.0 * (k - 1) * h_nm2[j]
            h_nm2, h_nm1 = h_nm1, h_n
        return h_nm1

_H0_c = _hermite_coeffs(0)   # [1]
_H4_c = _hermite_coeffs(4)   # [3, 0, -6, 0, 1]
_H8_c = _hermite_coeffs(8)
_H12_c = _hermite_coeffs(12)

# Constraint: g(0) = 0 means P(0) = 0
# P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
# So: a0 + 3*a4 + 105*a8 + 945*a12 = 0
# a12 = -(a0 + 3*a4 + 105*a8) / 945

def _get_polynomial_coeffs(a0, a4, a8):
    """
    Get polynomial coefficients p_k where P(x) = sum p_k x^(2k)
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12, with a12 determined by P(0)=0.
    Returns array of length 7 for x^0, x^2, x^4, ..., x^12.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Build P(x) = sum_{k=0}^{12} p_k x^k
    # Then extract even powers only
    p_full = np.zeros(13)
    
    for coeff, h_coeffs in [(a0, _H0_c), (a4, _H4_c), (a8, _H8_c), (a12, _H12_c)]:
        for k, c in enumerate(h_coeffs):
            p_full[k] += coeff * c
    
    # Extract even powers: p[0]=x^0, p[1]=x^2, ..., p[6]=x^12
    p_even = np.zeros(7)
    for k in range(7):
        p_even[k] = p_full[2 * k]
    
    return p_even


def _compute_bound_exact(a0, a4, a8):
    """
    Compute the uncertainty product bound exactly using moment integrals.
    
    g(x) = P(x) * exp(-x^2/2)
    g'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    
    The Fourier transform of g involves Q(x) where Q_k = (-1)^k * p_k
    (for even polynomial, the FT of x^(2k)*exp(-x^2/2) relates to derivatives)
    
    Actually, let's use the variance formula:
    A(f) = ||f||_2 * sqrt(<x^2> - <x>^2) ... but for this specific form:
    
    The bound is |integral g| * ||g'||_2 * ||G||_2 / (||g||_2)^2 * ||g||_2
    Wait, let me reconsider.
    
    A(f)*A(F) where A(f) = ||f'||_2 / ||f||_2 * ||f||_2 = ||f'||_2
    No: A(f) = ||f'||_2 (L2 norm of derivative)
    A(F) = ||x F||_2 (L2 norm of x times FT)
    
    By Plancherel: ||x F||_2 = ||(FT)^{-1}(d/dx g)||_2 ... 
    Actually ||x F||_2^2 = ||(i d/dxi F)||_2^2 ... 
    
    Let me use: ||x F||_2^2 = 2*pi * ||g'||_2^2 (for appropriate normalization)
    No, that's not right either.
    
    Let me just compute directly:
    - ||g||_2^2 = integral |g|^2 dx = integral P(x)^2 exp(-x^2) dx
    - ||g'||_2^2 = integral |g'(x)|^2 dx where g'(x) = (P'(x) - xP(x)) exp(-x^2/2)
    - ||G||_2^2 = integral |G(omega)|^2 domega = 2*pi * ||g||_2^2 (Plancherel with our FT convention)
    - ||omega G||_2^2 = 2*pi * ||g'||_2^2 (since FT of g' is i*omega*G)
    
    So the bound = |integral g| * ||g'||_2 * ||omega G||_2 / ||g||_2
    Wait, A(f)*A(F) = ||f'||_2 * ||omega F||_2
    
    And by Plancherel (unitary FT): ||f||_2 = ||F||_2
    ||f'||_2 = ||i omega F||_2 = ||omega F||_2
    
    So A(f)*A(F) = ||f'||_2 * ||omega F||_2 = ||f'||_2^2 = ||g'||_2^2
    
    Hmm, that would make it just ||g'||_2^2 which doesn't depend on the Fourier side separately.
    
    Let me re-read the task: "upper bound on A(f)*A(Fourier(f))"
    A(f) is likely the uncertainty: A(f) = sqrt(<x^2> - <x>^2) * ||f|| / ||f|| ... 
    Actually A(f) = ||x f||_2 / ||f||_2 (position uncertainty)
    A(F) = ||omega F||_2 / ||F||_2 (frequency uncertainty)
    
    Then A(f)*A(F) = (||x f||_2 * ||omega F||_2) / ||f||_2^2
    
    By Plancherel: ||omega F||_2 = ||f'||_2 (with proper constants)
    And ||x f||_2^2 = integral x^2 |f|^2 dx
    
    So bound = ||x g||_2 * ||g'||_2 / ||g||_2
    
    Let me compute all three using exact moments.
    """
    p = _get_polynomial_coeffs(a0, a4, a8)
    # p[k] is coefficient of x^(2k) in P(x)
    # P(x) = p[0] + p[1]x^2 + p[2]x^4 + ... + p[6]x^12
    
    # g(x) = P(x) * exp(-x^2/2)
    # g'(x) = (P'(x) - x*P(x)) * exp(-x^2/2)
    # P'(x) = 2*p[1]*x + 4*p[2]*x^3 + ... + 24*p[6]*x^11
    # x*P(x) = p[0]*x + p[1]*x^3 + ... + p[6]*x^13
    # P'(x) - x*P(x) = (2p[1]-p[0])*x + (4p[2]-p[1])*x^3 + ... + (24p[6]-p[5])*x^11 - p[6]*x^13
    
    # Let Q(x) = P'(x) - x*P(x) = sum_{k=0}^{6} q_k x^(2k+1) + q_7 x^13
    # where q_k = (2(k+1)*p[k+1] - p[k]) for k=0..5, and q_6 = 24*p[6] - p[5], plus -p[6]*x^13
    
    # Actually: Q(x) = sum_{k=0}^{6} (2(k+1)*p[k+1] - p[k]) * x^(2k+1)  [for k where k+1<=6]
    # For k=6: term is 2*7*p[7] - p[6] but p[7]=0, so just -p[6]*x^13
    
    # Let me define q_odd[j] = coefficient of x^(2j+1) in Q(x), j=0..6
    q_odd = np.zeros(7)
    for j in range(6):
        q_odd[j] = 2.0 * (j + 1) * p[j + 1] - p[j]
    q_odd[6] = -p[6]  # from -x*P(x) term, P'(x) has no x^13 term
    
    # Now compute integrals:
    # ||g||_2^2 = integral P(x)^2 exp(-x^2) dx
    # P(x)^2 = (sum_i p_i x^(2i))^2 = sum_{i,j} p_i p_j x^(2(i+j))
    # integral x^(2n) exp(-x^2) dx = I[n]
    g_norm_sq = 0.0
    for i in range(7):
        for j in range(7):
            n = i + j
            if n < _N_MOMENTS:
                g_norm_sq += p[i] * p[j] * _I[n]
    
    # ||g'||_2^2 = integral Q(x)^2 exp(-x^2) dx
    # Q(x) = sum_j q_odd[j] x^(2j+1)
    # Q(x)^2 = sum_{j,k} q_odd[j] q_odd[k] x^(2(j+k+1))
    gp_norm_sq = 0.0
    for j in range(7):
        for k in range(7):
            n = j + k + 1
            if n < _N_MOMENTS:
                gp_norm_sq += q_odd[j] * q_odd[k] * _I[n]
    
    # ||x g||_2^2 = integral x^2 P(x)^2 exp(-x^2) dx
    # = integral sum_{i,j} p_i p_j x^(2(i+j+1)) exp(-x^2) dx
    xg_norm_sq = 0.0
    for i in range(7):
        for j in range(7):
            n = i + j + 1
            if n < _N_MOMENTS:
                xg_norm_sq += p[i] * p[j] * _I[n]
    
    # integral g dx = integral P(x) exp(-x^2/2) dx
    # = sum_i p_i * integral x^(2i) exp(-x^2/2) dx
    # integral x^(2i) exp(-x^2/2) dx = 2^(i+0.5) * sqrt(pi) * (2i-1)!! / 2^i ... 
    # Actually: integral x^(2n) exp(-x^2/2) dx = 2^n * n! * sqrt(2*pi) / 2^n ... 
    # Let J[n] = integral x^(2n) exp(-x^2/2) dx
    # J[0] = sqrt(2*pi)
    # J[n] = (2n-1) * J[n-1]
    J = np.zeros(7)
    J[0] = np.sqrt(2.0 * np.pi)
    for n in range(1, 7):
        J[n] = (2.0 * n - 1.0) * J[n - 1]
    
    int_g = 0.0
    for i in range(7):
        int_g += p[i] * J[i]
    
    # Bound = |int_g| * sqrt(gp_norm_sq) * sqrt(xg_norm_sq) / sqrt(g_norm_sq)
    # Wait: A(f)*A(F) = (||xg||_2/||g||_2) * (||g'||_2/||g||_2) = ||xg||_2 * ||g'||_2 / ||g||_2^2
    
    if g_norm_sq <= 0 or gp_norm_sq <= 0 or xg_norm_sq <= 0:
        return 1e18
    if not np.isfinite(g_norm_sq) or not np.isfinite(gp_norm_sq) or not np.isfinite(xg_norm_sq):
        return 1e18
    
    bound = np.sqrt(xg_norm_sq) * np.sqrt(gp_norm_sq) / g_norm_sq
    
    if not np.isfinite(bound) or bound <= 0:
        return 1e18
    
    return bound


def _objective_and_grad(coeffs):
    """Return (value, grad) for the upper bound C4 given (a0,a4,a8)."""
    a0, a4, a8 = coeffs
    val = _compute_bound_exact(a0, a4, a8)
    
    # Numerical gradient (central differences)
    h = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        cp = coeffs.copy()
        cm = coeffs.copy()
        cp[i] += h
        cm[i] -= h
        fp = _compute_bound_exact(cp[0], cp[1], cp[2])
        fm = _compute_bound_exact(cm[0], cm[1], cm[2])
        grad[i] = (fp - fm) / (2.0 * h)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(3)
    
    return val, grad


def construct():
    """Return the three Hermite coefficients (degrees 0,4,8)."""
    # Prior state of the art as a starting point.
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    try:
        val0 = _compute_bound_exact(best_coeffs[0], best_coeffs[1], best_coeffs[2])
        best_val = val0
    except Exception:
        return best_coeffs
    
    if not np.isfinite(best_val) or best_val >= 1e17:
        # Try to find a valid starting point
        for trial in range(100):
            tc = np.array([
                -1.0 + 0.5 * np.random.uniform(-1, 1),
                0.04 + 0.02 * np.random.uniform(-1, 1),
                0.0003 + 0.0002 * np.random.uniform(-1, 1)
            ])
            tv = _compute_bound_exact(tc[0], tc[1], tc[2])
            if np.isfinite(tv) and tv < best_val:
                best_val = tv
                best_coeffs = tc.copy()
                break
    
    # Multi-start L-BFGS-B
    rng = np.random.default_rng(42)
    starts = [best_coeffs.copy()]
    
    # Add diverse starts
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        starts.append(best_coeffs + rng.normal(0, scale, size=3))
    
    # Structured starts
    for delta in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
        starts.append(best_coeffs + np.array([delta, delta * 0.5, delta * 0.25]))
    
    # Wider exploration
    for _ in range(20):
        s = np.array([
            rng.uniform(-3.0, 2.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.01, 0.01)
        ])
        starts.append(s)
    
    # Evaluate and sort starts
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            v = _compute_bound_exact(s[0], s[1], s[2])
            if np.isfinite(v):
                scored.append((v, s.copy()))
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from top starts
    for _, s in scored[:10]:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(
                _objective_and_grad,
                s,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # High-precision polish
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                _objective_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish for very tight convergence
    if time.time() < _DEADLINE - 5:
        try:
            def _obj_only(x):
                return _compute_bound_exact(x[0], x[1], x[2])
            
            nm_res = minimize(
                _obj_only,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True},
            )
            if np.isfinite(nm_res.fun) and nm_res.fun < best_val:
                best_val = nm_res.fun
                best_coeffs = nm_res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        return np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    return best_coeffs
# EVOLVE-BLOCK-END
