# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
import time


def _build_quadrature():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-point phi grid."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    weights = np.outer(w_gl, np.full(n_phi, dphi))
    cos_theta = x_gl
    return cos_theta, phi, weights


def _spherical_harmonics_real(cos_theta, phi, k=10):
    """
    Compute real orthonormal spherical harmonics of degree k.
    Returns array of shape (2*k+1, n_theta, n_phi) where index i corresponds to m = i - k.
    """
    n_theta = len(cos_theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_abs in range(k + 1):
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * 
                      float(factorial(k - m_abs) / factorial(k + m_abs)))
        P_m = lpmv(m_abs, k, cos_theta)
        
        if m_abs == 0:
            Y[0] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 using precomputed harmonics and weights."""
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R and its gradient with respect to coeffs (fully vectorized)."""
    n_modes = Y.shape[0]
    
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    f2 = f * f
    f4 = f2 * f2
    f3 = f * f2
    
    w_f2 = weights * f2
    w_f4 = weights * f4
    w_f3 = weights * f3
    w_f = weights * f
    
    norm2_sq = np.sum(w_f2)
    norm4_sq = np.sum(w_f4)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros(n_modes)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    if norm4_sq <= 0:
        return R, np.zeros(n_modes)
    
    # Vectorized: d_norm4_sq_dc[m] = 4 * sum(w * f^3 * Y[m])
    d_norm4_sq_dc = 4.0 * np.tensordot(Y, w_f3, axes=([1, 2], [0, 1]))
    d_norm2_sq_dc = 2.0 * np.tensordot(Y, w_f, axes=([1, 2], [0, 1]))
    
    d_norm4_dc = (1.0 / (4.0 * norm4**3)) * d_norm4_sq_dc
    d_norm2_dc = (1.0 / (2.0 * norm2)) * d_norm2_sq_dc
    
    grad = d_norm4_dc / norm2 - norm4 * d_norm2_dc / (norm2 * norm2)
    
    return R, grad


def _random_unit_vector(rng, n=21):
    """Generate a random unit vector."""
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    return v


def _try_symmetric_patterns(Y, weights):
    """Try various symmetric coefficient patterns."""
    best_coeffs = None
    best_ratio = np.inf
    
    patterns = []
    
    # All ones normalized
    c = np.ones(21) / np.sqrt(21)
    patterns.append(c)
    
    # Only m=0
    c = np.zeros(21)
    c[10] = 1.0
    patterns.append(c)
    
    # m=0 and m=±10
    c = np.zeros(21)
    c[10] = 1.0
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Even m modes: 0, 2, 4, 6, 8, 10
    c = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Odd m modes: 1, 3, 5, 7, 9
    c = np.zeros(21)
    for m in [1, 3, 5, 7, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cosine-weighted
    c = np.zeros(21)
    for m in range(11):
        c[10 + m] = np.cos(m * np.pi / 20.0)
        if m > 0:
            c[10 - m] = np.cos(m * np.pi / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Sine-weighted
    c = np.zeros(21)
    for m in range(1, 11):
        c[10 + m] = np.sin(m * np.pi / 21.0)
        c[10 - m] = np.sin(m * np.pi / 21.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # High-m concentration
    c = np.zeros(21)
    c[0] = 1.0
    c[20] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Low-m concentration
    c = np.zeros(21)
    c[10] = 1.0
    c[9] = 1.0
    c[11] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Alternating signs
    c = np.zeros(21)
    for m in range(21):
        c[m] = 1.0 if m % 2 == 0 else -1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Gaussian-like distribution
    c = np.zeros(21)
    for m in range(21):
        idx = m - 10
        c[m] = np.exp(-idx**2 / 20.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {4,5,6}
    c = np.zeros(21)
    for m in [4, 5, 6]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {7,8,9,10}
    c = np.zeros(21)
    for m in [7, 8, 9, 10]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {0,1,2}
    c = np.zeros(21)
    for m in [0, 1, 2]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Sign-flipped cosine
    c = np.zeros(21)
    for m in range(11):
        c[10 + m] = np.cos(m * np.pi / 20.0) * (1.0 if m % 2 == 0 else -1.0)
        if m > 0:
            c[10 - m] = np.cos(m * np.pi / 20.0) * (1.0 if m % 2 == 0 else -1.0)
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Quadratic in m
    c = np.zeros(21)
    for m in range(11):
        w = 1.0 - m * m / 121.0
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Power-law weighting
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        c = np.zeros(21)
        for m in range(11):
            w = (m + 1) ** p
            c[10 + m] = w
            if m > 0:
                c[10 - m] = w
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Inverse power-law
    for p in [0.5, 1.0, 1.5, 2.0]:
        c = np.zeros(21)
        for m in range(11):
            w = 1.0 / (m + 1) ** p
            c[10 + m] = w
            if m > 0:
                c[10 - m] = w
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Cosine with phase shift
    for phase in [0.1, 0.2, 0.3, 0.4, 0.5]:
        c = np.zeros(21)
        for m in range(11):
            w = np.cos(m * np.pi / 20.0 + phase)
            c[10 + m] = w
            if m > 0:
                c[10 - m] = w
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Bessel-like
    c = np.zeros(21)
    for m in range(11):
        w = np.sin(m * 0.5) / (m + 1)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Alternating with decay
    c = np.zeros(21)
    for m in range(11):
        w = np.exp(-m * m / 15.0) * (1.0 if m % 2 == 0 else -0.5)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Broader Gaussian
    for sigma in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        c = np.zeros(21)
        for m in range(21):
            idx = m - 10
            c[m] = np.exp(-idx**2 / (2.0 * sigma**2))
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Triangular in |m|
    c = np.zeros(21)
    for m in range(11):
        w = max(0, 1.0 - m / 10.0)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Peak at m=5
    c = np.zeros(21)
    for m in range(11):
        w = np.exp(-(m - 5)**2 / 8.0)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Peak at m=3
    c = np.zeros(21)
    for m in range(11):
        w = np.exp(-(m - 3)**2 / 6.0)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Peak at m=7
    c = np.zeros(21)
    for m in range(11):
        w = np.exp(-(m - 7)**2 / 6.0)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Two peaks at m=2 and m=8
    c = np.zeros(21)
    for m in range(11):
        w = np.exp(-(m - 2)**2 / 4.0) + np.exp(-(m - 8)**2 / 4.0)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Two peaks at m=1 and m=9
    c = np.zeros(21)
    for m in range(11):
        w = np.exp(-(m - 1)**2 / 3.0) + np.exp(-(m - 9)**2 / 3.0)
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {3,4,5,6,7}
    c = np.zeros(21)
    for m in [3, 4, 5, 6, 7]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {5,6,7,8}
    c = np.zeros(21)
    for m in [5, 6, 7, 8]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {2,3,4}
    c = np.zeros(21)
    for m in [2, 3, 4]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {1,2,3,4}
    c = np.zeros(21)
    for m in [1, 2, 3, 4]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {6,7,8,9}
    c = np.zeros(21)
    for m in [6, 7, 8, 9]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {8,9,10}
    c = np.zeros(21)
    for m in [8, 9, 10]:
        c[10 + m] = 1.0
        c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {0,1,2,3}
    c = np.zeros(21)
    for m in [0, 1, 2, 3]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # m-segment: only |m| in {0,1}
    c = np.zeros(21)
    for m in [0, 1]:
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Linear ramp up
    c = np.zeros(21)
    for m in range(11):
        w = m / 10.0
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Linear ramp down
    c = np.zeros(21)
    for m in range(11):
        w = 1.0 - m / 10.0
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cubic in m
    c = np.zeros(21)
    for m in range(11):
        w = m**3 / 1000.0
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Inverse cubic
    c = np.zeros(21)
    for m in range(11):
        w = 1.0 / (m + 1)**3
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Cosine squared
    c = np.zeros(21)
    for m in range(11):
        w = np.cos(m * np.pi / 20.0)**2
        c[10 + m] = w
        if m > 0:
            c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Sine squared
    c = np.zeros(21)
    for m in range(1, 11):
        w = np.sin(m * np.pi / 22.0)**2
        c[10 + m] = w
        c[10 - m] = w
    c /= np.linalg.norm(c)
    patterns.append(c)
    
    # Random patterns with fixed seeds
    rng_pat = np.random.default_rng(12345)
    for _ in range(20):
        c = rng_pat.standard_normal(21)
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    # Symmetric random patterns
    rng_sym = np.random.default_rng(67890)
    for _ in range(15):
        c = np.zeros(21)
        for m in range(11):
            w = rng_sym.standard_normal()
            c[10 + m] = w
            if m > 0:
                c[10 - m] = w
        c /= np.linalg.norm(c)
        patterns.append(c)
    
    for c in patterns:
        r = _compute_ratio(c, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c.copy()
    
    return best_coeffs, best_ratio


def _simulated_annealing(c, Y, weights, rng, max_iter=3000, T_start=0.005, T_end=1e-7):
    """Simple simulated annealing on the sphere."""
    c = c.copy()
    c /= np.linalg.norm(c)
    best_c = c.copy()
    best_R = _compute_ratio(c, Y, weights)
    current_R = best_R
    
    for it in range(max_iter):
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Random perturbation with adaptive scale
        scale = 0.05 + 0.05 * (T / T_start)
        delta = rng.standard_normal(21) * scale
        c_new = c + delta
        c_new /= np.linalg.norm(c_new)
        
        R_new = _compute_ratio(c_new, Y, weights)
        
        if R_new < best_R:
            best_R = R_new
            best_c = c_new.copy()
            current_R = R_new
        else:
            # Accept with probability
            if rng.random() < np.exp(-(R_new - current_R) / T):
                c = c_new
                current_R = R_new
    
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 90.0  # seconds
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Try symmetric patterns first
    best_coeffs, best_ratio = _try_symmetric_patterns(Y, weights)
    
    if best_coeffs is None:
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    # Set up optimization
    rng = np.random.default_rng(42)
    
    def objective_with_grad(c):
        if time.time() > deadline - safety_margin:
            return _compute_ratio(best_coeffs, Y, weights), np.zeros(21)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros(21)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        # Project gradient to constraint surface
        grad_proj = grad - np.dot(grad, c_normalized) * c_normalized
        return R, grad_proj
    
    # Phase 1: Broad search with many diverse restarts
    n_restarts_phase1 = 120
    
    for i in range(n_restarts_phase1):
        if time.time() > deadline - safety_margin:
            break
        
        # Choose initial guess
        if i == 0:
            c_init = best_coeffs.copy()
        elif i < 35:
            # Structured starts - more diverse
            if i == 1:
                c_init = np.zeros(21)
                c_init[10] = 1.0
            elif i == 2:
                c_init = np.zeros(21)
                c_init[20] = 1.0
            elif i == 3:
                c_init = np.zeros(21)
                c_init[0] = 1.0
            elif i == 4:
                c_init = np.zeros(21)
                c_init[10:16] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 5:
                c_init = np.zeros(21)
                c_init[10:14] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 6:
                c_init = np.zeros(21)
                c_init[10:12] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 7:
                c_init = np.zeros(21)
                c_init[10:18] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 8:
                c_init = np.zeros(21)
                c_init[10:21] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 9:
                c_init = np.zeros(21)
                for m in range(0, 11, 2):
                    c_init[10 + m] = 1.0
                    if m > 0:
                        c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 10:
                c_init = np.zeros(21)
                for m in [4, 5, 6]:
                    c_init[10 + m] = 1.0
                    c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 11:
                c_init = np.zeros(21)
                for m in [7, 8, 9, 10]:
                    c_init[10 + m] = 1.0
                    c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 12:
                c_init = np.zeros(21)
                for m in [0, 1, 2]:
                    c_init[10 + m] = 1.0
                    if m > 0:
                        c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 13:
                c_init = np.zeros(21)
                for m in range(11):
                    c_init[10 + m] = np.cos(m * np.pi / 20.0) * (1.0 if m % 2 == 0 else -1.0)
                    if m > 0:
                        c_init[10 - m] = np.cos(m * np.pi / 20.0) * (1.0 if m % 2 == 0 else -1.0)
                c_init /= np.linalg.norm(c_init)
            elif i == 14:
                c_init = np.zeros(21)
                for m in range(11):
                    w = 1.0 - m * m / 121.0
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 15:
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-m * m / 30.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 16:
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-m * m / 8.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 17:
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-m * m / 50.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 18:
                c_init = np.zeros(21)
                for m in range(21):
                    c_init[m] = 1.0 if (m - 10) % 3 == 0 else -0.5
                c_init /= np.linalg.norm(c_init)
            elif i == 19:
                c_init = np.zeros(21)
                for m in range(21):
                    c_init[m] = np.sin(m * np.pi / 10.5)
                c_init /= np.linalg.norm(c_init)
            elif i == 20:
                # Peak at m=3
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 3)**2 / 10.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 21:
                # Peak at m=7
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 7)**2 / 10.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 22:
                # Two peaks
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 2)**2 / 5.0) + np.exp(-(m - 8)**2 / 5.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 23:
                # Linear ramp
                c_init = np.zeros(21)
                for m in range(11):
                    w = m / 10.0
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 24:
                # Inverse linear ramp
                c_init = np.zeros(21)
                for m in range(11):
                    w = 1.0 - m / 10.0
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 25:
                # Peak at m=4
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 4)**2 / 6.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 26:
                # Peak at m=6
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 6)**2 / 6.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 27:
                # Two peaks at m=3 and m=7
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 3)**2 / 4.0) + np.exp(-(m - 7)**2 / 4.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 28:
                # Three peaks
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.exp(-(m - 1)**2 / 3.0) + np.exp(-(m - 5)**2 / 3.0) + np.exp(-(m - 9)**2 / 3.0)
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 29:
                # m-segment {2,3,4,5,6}
                c_init = np.zeros(21)
                for m in [2, 3, 4, 5, 6]:
                    c_init[10 + m] = 1.0
                    c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 30:
                # m-segment {4,5,6,7,8}
                c_init = np.zeros(21)
                for m in [4, 5, 6, 7, 8]:
                    c_init[10 + m] = 1.0
                    c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 31:
                # m-segment {6,7,8,9,10}
                c_init = np.zeros(21)
                for m in [6, 7, 8, 9, 10]:
                    c_init[10 + m] = 1.0
                    c_init[10 - m] = 1.0
                c_init /= np.linalg.norm(c_init)
            elif i == 32:
                # Cubic weighting
                c_init = np.zeros(21)
                for m in range(11):
                    w = m**3 / 1000.0
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            elif i == 33:
                # Cosine squared
                c_init = np.zeros(21)
                for m in range(11):
                    w = np.cos(m * np.pi / 20.0)**2
                    c_init[10 + m] = w
                    if m > 0:
                        c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
            else:
                # Sine squared
                c_init = np.zeros(21)
                for m in range(1, 11):
                    w = np.sin(m * np.pi / 22.0)**2
                    c_init[10 + m] = w
                    c_init[10 - m] = w
                c_init /= np.linalg.norm(c_init)
        else:
            c_init = _random_unit_vector(rng)
        
        # Run L-BFGS-B with gradient
        result = minimize(
            objective_with_grad,
            c_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Phase 2: Simulated annealing from best found
    if time.time() < deadline - safety_margin:
        c_sa, R_sa = _simulated_annealing(best_coeffs, Y, weights, rng, max_iter=4000)
        if R_sa < best_ratio:
            best_ratio = R_sa
            best_coeffs = c_sa
    
    # Phase 3: Focused refinement around best found
    if time.time() < deadline - safety_margin:
        for i in range(20):
            if time.time() > deadline - safety_margin:
                break
            
            c_init = best_coeffs + 0.05 * rng.standard_normal(21)
            c_init /= np.linalg.norm(c_init)
            
            result = minimize(
                objective_with_grad,
                c_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 150}
            )
            
            if result.fun < best_ratio:
                c_opt = result.x / np.linalg.norm(result.x)
                r_opt = _compute_ratio(c_opt, Y, weights)
                if r_opt < best_ratio:
                    best_ratio = r_opt
                    best_coeffs = c_opt
    
    # Phase 4: Coordinate descent polish
    if time.time() < deadline - safety_margin:
        c = best_coeffs.copy()
        for _ in range(100):
            if time.time() > deadline - safety_margin:
                break
            improved = False
            for m in range(21):
                if time.time() > deadline - safety_margin:
                    break
                # Try perturbing component m with fine steps
                for delta in [-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]:
                    c_new = c.copy()
                    c_new[m] += delta
                    c_new /= np.linalg.norm(c_new)
                    r_new = _compute_ratio(c_new, Y, weights)
                    if r_new < best_ratio - 1e-14:
                        c = c_new
                        best_ratio = r_new
                        improved = True
                        break
            if not improved:
                break
        best_coeffs = c
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - safety_margin:
        # Steepest descent on the sphere with adaptive step
        c = best_coeffs.copy()
        for it in range(200):
            if time.time() > deadline - safety_margin:
                break
            R, grad = _compute_ratio_and_grad(c, Y, weights)
            grad_proj = grad - np.dot(grad, c) * c
            gnorm = np.linalg.norm(grad_proj)
            if gnorm < 1e-14:
                break
            # Adaptive step size via backtracking
            step = 0.5
            for _ in range(40):
                c_new = c - step * grad_proj
                c_new /= np.linalg.norm(c_new)
                R_new = _compute_ratio(c_new, Y, weights)
                if R_new < R - 1e-16:
                    c = c_new
                    break
                step *= 0.5
            if step < 1e-12:
                break
        best_coeffs = c
        best_ratio = _compute_ratio(c, Y, weights)
        
        # Final L-BFGS-B polish with very tight tolerances
        if time.time() < deadline - safety_margin:
            result = minimize(
                objective_with_grad,
                best_coeffs,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200}
            )
            if result.fun < best_ratio:
                c_opt = result.x / np.linalg.norm(result.x)
                r_opt = _compute_ratio(c_opt, Y, weights)
                if r_opt < best_ratio:
                    best_ratio = r_opt
                    best_coeffs = c_opt
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
