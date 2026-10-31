"""Optimized uncertainty bound via analytic Hermite-Gaussian moments + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _hermite_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 (even polynomials in x)."""
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    h0 = np.array([1.0])           # coeff of x^0
    h4 = np.array([3.0, -6.0, 1.0])  # coeff of x^0, x^2, x^4
    h8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0])  # x^0, x^2, x^4, x^6, x^8
    h12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])  # x^0..x^12
    return h0, h4, h8, h12


def _poly_sq_moments(coeffs):
    """
    Given a polynomial P(x) = sum_k coeffs[k] * x^(2k) (even powers only),
    compute the moments M_{2n} = integral x^{2n} P(x)^2 e^{-x^2} dx for n=0,1,2.
    Uses: integral x^{2m} e^{-x^2} dx = sqrt(pi) * (2m-1)!! / 2^m
    """
    # P(x)^2 = sum_{j,k} coeffs[j]*coeffs[k] * x^{2(j+k)}
    # M_{2n} = sum_{j,k} coeffs[j]*coeffs[k] * integral x^{2n + 2(j+k)} e^{-x^2} dx
    #        = sum_{j,k} coeffs[j]*coeffs[k] * I(2n + 2(j+k))
    # where I(2m) = sqrt(pi) * (2m-1)!! / 2^m

    sqrt_pi = np.sqrt(np.pi)

    def I_2m(m):
        """integral x^{2m} e^{-x^2} dx = sqrt(pi) * (2m-1)!! / 2^m"""
        if m == 0:
            return sqrt_pi
        # (2m-1)!! = product of odd numbers from 1 to 2m-1
        result = 1.0
        for i in range(1, 2 * m, 2):
            result *= i
        return sqrt_pi * result / (2.0 ** m)

    nc = len(coeffs)
    M = np.zeros(3)  # M_0, M_2, M_4
    for n in range(3):
        s = 0.0
        for j in range(nc):
            for k in range(nc):
                m = n + j + k  # power is 2m
                s += coeffs[j] * coeffs[k] * I_2m(m)
        M[n] = s
    return M


def _fourier_poly_coeffs(coeffs):
    """
    For g(x) = P(x) e^{-x^2/2}, the Fourier transform (with F[f](xi) = int f(x) e^{-ixxi} dx)
    gives G(xi) proportional to P(-i*xi) e^{-xi^2/2} (up to a constant factor that cancels in ratios).
    
    P(-i*xi) where P(x) = sum_k c_k x^{2k}:
    P(-i*xi) = sum_k c_k (-i)^{2k} xi^{2k} = sum_k c_k (-1)^k xi^{2k}
    
    So the polynomial in xi has coefficients c_k * (-1)^k.
    """
    return coeffs * ((-1.0) ** np.arange(len(coeffs)))


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product A(g)*A(G) for given coefficients."""
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # P(0) = a0*h0[0] + a4*h4[0] + a8*h8[0] + a12*h12[0] = 0
    # h0[0]=1, h4[0]=3, h8[0]=105, h12[0]=945
    # a12 = -(a0 + 3*a4 + 105*a8) / 945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Build P(x) as coefficient array for x^0, x^2, ..., x^12
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Each H has even powers, so combine:
    nc = 7  # powers x^0, x^2, ..., x^12
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
    
    # Fourier side: Q(xi) = P(-i*xi), coefficients are P_k * (-1)^k
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
    
    # A(f) = sqrt(<x^4> - <x^2>^2) = sqrt(dx2)
    # A(F(f)) = sqrt(<xi^4> - <xi^2>^2) = sqrt(dxi2)
    # Product = sqrt(dx2 * dxi2)
    product = np.sqrt(dx2 * dxi2)
    return product


def _objective(params):
    """Objective function for optimizer."""
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)


def _objective_grad(params):
    """Analytic gradient via finite differences (fast enough with analytic objective)."""
    eps = 1e-7
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
    
    # Best known starting point (Goncalves et al. 2017)
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Multi-start L-BFGS-B
    starts = [
        x0,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.06, 0.0005]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0002]),
        np.array([-0.8, 0.05, 0.0004]),
        np.array([-1.5, 0.02, 0.0001]),
    ]
    
    for i, x_start in enumerate(starts):
        if time.time() - t0 > 180:
            break
        try:
            res = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance from best point
    if time.time() - t0 < 190:
        try:
            res = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional refinement with Nelder-Mead from best point
    if time.time() - t0 < 195:
        try:
            res = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
