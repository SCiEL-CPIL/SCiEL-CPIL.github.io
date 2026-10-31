"""Optimized uncertainty bound via analytic Hermite-Gaussian moments + analytic gradient + global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 195  # 15s safety margin before 210s kill


def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 (even polynomials in x)."""
    # Physicists' Hermite polynomials (even ones):
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    h0 = np.array([1.0])
    h4 = np.array([3.0, -6.0, 1.0])
    h8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0])
    h12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    return h0, h4, h8, h12


def _I_2m(m):
    """integral x^{2m} e^{-x^2} dx = sqrt(pi) * (2m-1)!! / 2^m"""
    if m == 0:
        return np.sqrt(np.pi)
    # (2m-1)!! = product of odd numbers from 1 to 2m-1
    result = 1.0
    for i in range(1, 2 * m, 2):
        result *= i
    return np.sqrt(np.pi) * result / (2.0 ** m)


# Precompute I_2m for m=0..12 (max needed: n=2, j+k up to 6, so m up to 8)
# Actually m = n + j + k where n in {0,1,2}, j,k in 0..6, so m up to 2+6+6=14
_I_TABLE = np.array([_I_2m(m) for m in range(15)])


def _poly_sq_moments(coeffs):
    """
    Given a polynomial P(x) = sum_k coeffs[k] * x^(2k) (even powers only),
    compute the moments M_{2n} = integral x^{2n} P(x)^2 e^{-x^2} dx for n=0,1,2.
    """
    nc = len(coeffs)
    M = np.zeros(3)
    for n in range(3):
        s = 0.0
        for j in range(nc):
            for k in range(nc):
                m = n + j + k
                s += coeffs[j] * coeffs[k] * _I_TABLE[m]
        M[n] = s
    return M


def _fourier_poly_coeffs(coeffs):
    """
    For g(x) = P(x) e^{-x^2/2}, the Fourier transform gives G(xi) proportional to
    P(-i*xi) e^{-xi^2/2}. P(-i*xi) has coefficients c_k * (-1)^k.
    """
    return coeffs * ((-1.0) ** np.arange(len(coeffs)))


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product A(g)*A(G) for given coefficients."""
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # P(0) = a0*h0[0] + a4*h4[0] + a8*h8[0] + a12*h12[0] = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Build P(x) as coefficient array for x^0, x^2, ..., x^12
    nc = 7
    P = np.zeros(nc)
    P += a0 * np.pad(h0, (0, nc - len(h0)))
    P += a4 * np.pad(h4, (0, nc - len(h4)))
    P += a8 * np.pad(h8, (0, nc - len(h8)))
    P += a12 * np.pad(h12, (0, nc - len(h12)))
    
    # Spatial moments
    Mx = _poly_sq_moments(P)
    m0_x, m2_x, m4_x = Mx[0], Mx[1], Mx[2]
    
    if m0_x <= 0:
        return 1e10
    
    ex2 = m2_x / m0_x
    ex4 = m4_x / m0_x
    dx2 = ex4 - ex2 * ex2
    if dx2 <= 0:
        return 1e10
    
    # Fourier side
    Q = _fourier_poly_coeffs(P)
    Mxi = _poly_sq_moments(Q)
    m0_xi, m2_xi, m4_xi = Mxi[0], Mxi[1], Mxi[2]
    
    if m0_xi <= 0:
        return 1e10
    
    exi2 = m2_xi / m0_xi
    exi4 = m4_xi / m0_xi
    dxi2 = exi4 - exi2 * exi2
    if dxi2 <= 0:
        return 1e10
    
    product = np.sqrt(dx2 * dxi2)
    return product


def _objective(params):
    """Objective function for optimizer."""
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)


def _objective_grad(params):
    """Analytic gradient via central finite differences."""
    eps = 1e-8
    grad = np.zeros(3)
    for i in range(3):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (_objective(p_plus) - _objective(p_minus)) / (2 * eps)
    return grad


def construct():
    """Return three coefficients (a0, a4, a8) minimizing the uncertainty bound."""
    t0 = time.time()
    
    # Known good starting point (Goncalves et al. 2017)
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Phase 1: Differential evolution for global search (broad exploration)
    if time.time() - t0 < 120:
        try:
            bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_res = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                tol=1e-12,
                popsize=15,
                seed=42,
                polish=True,
                updating='deferred',
                init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with analytic gradient from multiple starts
    starts = [
        best_x,
        x0,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.06, 0.0005]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0004]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-1.3, 0.045, 0.00035]),
        np.array([-0.9, 0.035, 0.00025]),
    ]
    
    for i, x_start in enumerate(starts):
        if time.time() - t0 > 175:
            break
        try:
            res = minimize(
                _objective,
                x_start,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Polish with even tighter tolerance
    if time.time() - t0 < 190:
        try:
            res = minimize(
                _objective,
                best_x,
                jac=_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 200}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead refinement
    if time.time() - t0 < 195:
        try:
            res = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
