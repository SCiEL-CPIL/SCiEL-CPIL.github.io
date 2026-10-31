import numpy as np
import time
from scipy.optimize import minimize, brent

# Precompute J_m = ∫ x^{2m} e^{-x²} dx = (2m)!/(4^m m!) √π
def _precompute_J(max_m=14):
    J = np.zeros(max_m + 1)
    for m in range(max_m + 1):
        # (2m)! / (4^m m!) * sqrt(pi)
        log_val = 0.0
        for k in range(1, 2*m + 1):
            log_val += np.log(k)
        log_val -= m * np.log(4.0)
        for k in range(1, m + 1):
            log_val -= np.log(k)
        log_val += 0.5 * np.log(np.pi)
        J[m] = np.exp(log_val)
    return J

_J = _precompute_J(14)

# Hermite polynomial coefficients in basis {1, x², x⁴, ..., x¹²}
# H_0 = 1
# H_4 = 3 - 6x² + x⁴
# H_8 = 105 - 420x² + 210x⁴ - 28x⁶ + x⁸
# H_12 = 945 - 7560x² + 13860x⁴ - 6930x⁶ + 1155x⁸ - 66x¹⁰ + x¹²
H_coeffs = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [3, -6, 1, 0, 0, 0, 0],
    [105, -420, 210, -28, 1, 0, 0],
    [945, -7560, 13860, -6930, 1155, -66, 1]
], dtype=float)

# H_n(0) values
H0_at_0 = 1.0
H4_at_0 = 3.0
H8_at_0 = 105.0
H12_at_0 = 945.0


def _compute_bound_and_grad(c):
    """
    Compute the uncertainty bound sqrt(σ_x² · σ_ξ²) and its gradient.
    c = [c0, c4, c8], c12 derived from g(0)=0.
    """
    c0, c4, c8 = c
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    coeffs = np.array([c0, c4, c8, c12])
    
    # P(x) = Σ c_i H_i(x) = Σ_{k=0}^{6} p_k x^{2k}
    p = coeffs @ H_coeffs  # shape (7,)
    
    # Moments of |g|² = P(x)² e^{-x²}
    # m_k = ∫ x^{2k} P(x)² e^{-x²} dx = Σ_{i,j} p_i p_j J_{k+i+j}
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            m0 += pp * _J[i + j]
            m2 += pp * _J[i + j + 1]
    
    # Fourier transform: ĝ(ξ) = √(2π) P(-iξ) e^{-ξ²/2}
    # P(-iξ) = Σ p_k (-i)^{2k} ξ^{2k} = Σ p_k (-1)^k ξ^{2k}
    q = np.array([p[k] * ((-1.0)**k) for k in range(7)])
    
    # n_k = ∫ ξ^{2k} |ĝ|² dξ = 2π Σ_{i,j} q_i q_j J_{k+i+j}
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            n0 += qq * _J[i + j]
            n2 += qq * _J[i + j + 1]
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    
    sigma_x2 = m2 / m0
    sigma_xi2 = n2 / n0
    
    if sigma_x2 <= 0 or sigma_xi2 <= 0:
        return 1e18, np.zeros(3)
    
    bound = np.sqrt(sigma_x2 * sigma_xi2)
    
    # Gradient computation
    # dc12/dc = [-1/945, -3/945, -105/945]
    dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    for m_idx in range(3):
        # dp_k/dc_m = H_coeffs[m_idx, k] + H_coeffs[3, k] * dc12[m_idx]
        dp = H_coeffs[m_idx, :] + H_coeffs[3, :] * dc12[m_idx]
        
        # dm0/dc = 2 Σ_i dp_i Σ_j p_j J_{i+j}
        # dm2/dc = 2 Σ_i dp_i Σ_j p_j J_{i+j+1}
        dm0 = 0.0
        dm2 = 0.0
        for i in range(7):
            dpi = dp[i]
            if abs(dpi) < 1e-18:
                continue
            for j in range(7):
                dm0 += 2.0 * dpi * p[j] * _J[i + j]
                dm2 += 2.0 * dpi * p[j] * _J[i + j + 1]
        
        # dq_k/dc_m = dp_k * (-1)^k
        dq = np.array([dp[k] * ((-1.0)**k) for k in range(7)])
        
        dn0 = 0.0
        dn2 = 0.0
        for i in range(7):
            dqi = dq[i]
            if abs(dqi) < 1e-18:
                continue
            for j in range(7):
                dn0 += 2.0 * 2.0*np.pi * dqi * q[j] * _J[i + j]
                dn2 += 2.0 * 2.0*np.pi * dqi * q[j] * _J[i + j + 1]
        
        d_sigma_x2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        d_sigma_xi2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        # d(sqrt(sx2 * sxi2))/dc = (sxi2 * d_sx2 + sx2 * d_sxi2) / (2 * bound)
        grad[m_idx] = (sigma_xi2 * d_sigma_x2 + sigma_x2 * d_sigma_xi2) / (2.0 * bound)
    
    return bound, grad


def _objective_and_grad(c):
    val, grad = _compute_bound_and_grad(c)
    return val, grad


def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin for 180s limit
    
    best_val = float('inf')
    best_c = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Strategy 1: 2D grid search over ratios (c4/c0, c8/c0) with c0=1
    # This exploits scale invariance
    grid_range1 = np.linspace(-3.0, 3.0, 40)  # c4/c0
    grid_range2 = np.linspace(-0.1, 0.3, 40)   # c8/c0
    
    for r1 in grid_range1:
        if time.time() > deadline - 30:
            break
        for r2 in grid_range2:
            c = np.array([1.0, r1, r2])
            try:
                val, _ = _compute_bound_and_grad(c)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_c = c.copy()
            except Exception:
                pass
    
    # Strategy 2: Local refinement from best grid point
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _objective_and_grad, best_c,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Multi-start with diverse initial guesses
    initial_guesses = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([1.0, -0.5, 0.05]),
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.5, -0.2, 0.02]),
        np.array([-2.0, 0.3, 0.03]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.1]),
        np.array([0.0, -1.0, 0.1]),
    ]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(15):
        initial_guesses.append(rng.normal(0, 1.0, 3))
    
    for guess in initial_guesses:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(
                _objective_and_grad, guess,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_and_grad, best_c,
                method='L-BFGS-B', jac=True,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return best_c
