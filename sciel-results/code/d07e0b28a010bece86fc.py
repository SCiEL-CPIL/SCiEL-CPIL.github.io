# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
from scipy.integrate import quad


def _hermite_combo(a0, a1, a2):
    """Return coefficients for H0, H4, H8, H12 with g(0)=0 (physicists' H)."""
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -2075040.0
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    return np.array([a0, a1, a2, a3])


def _poly_coeffs(coeffs):
    """Return coefficient array [c0, c1, ..., c12] for the polynomial P(x) = sum c_k x^k."""
    a0, a1, a2, a3 = coeffs
    # Build polynomial as sum of a_i * H_{4i}(x)
    # Use eval_hermite to get polynomial coefficients via roots or direct expansion
    # Simpler: evaluate on a grid and fit, but for exactness use known H_n polynomials
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 98304x^10 + 1032192x^8 - 5644800x^6 + 14155200x^4 - 15974400x^2 + 2075040
    c = np.zeros(13)
    # H0
    c[0] += a0 * 1.0
    # H4
    c[4] += a1 * 16.0
    c[2] += a1 * (-48.0)
    c[0] += a1 * 12.0
    # H8
    c[8] += a2 * 256.0
    c[6] += a2 * (-3072.0)
    c[4] += a2 * 11520.0
    c[2] += a2 * (-13440.0)
    c[0] += a2 * 1680.0
    # H12
    c[12] += a3 * 4096.0
    c[10] += a3 * (-98304.0)
    c[8] += a3 * 1032192.0
    c[6] += a3 * (-5644800.0)
    c[4] += a3 * 14155200.0
    c[2] += a3 * (-15974400.0)
    c[0] += a3 * 2075040.0
    return c


def _integrate_poly_gauss4(c, sign=1.0):
    """
    Compute integral of |P(x)|^4 * exp(-2x^2) dx over R.
    P(x) = sum c_k x^k. Since all c_k are real and only even powers appear,
    P(x) is real and even. So |P(x)|^4 = P(x)^4.
    We need int P(x)^4 exp(-2x^2) dx.
    P(x)^4 is a polynomial of degree 4*12=48, with only even powers.
    We can compute the coefficients of P^4 and integrate term by term.
    int x^{2m} exp(-2x^2) dx = 2^{-m-0.5} Gamma(m+0.5)
    """
    # Compute P^4 coefficients using convolution
    # P has only even powers, so let Q(t) = P(sqrt(t)) = sum c_{2j} t^j
    Q = np.array([c[2*j] for j in range(7)])  # degrees 0..6 in t
    # Q^4
    Q4 = Q
    for _ in range(3):
        Q4 = np.convolve(Q4, Q)
    # Q4[j] is coefficient of t^j in Q(t)^4 = P(x)^4 / ... wait
    # P(x) = sum c_{2j} x^{2j} = Q(x^2)
    # P(x)^4 = Q(x^2)^4 = Q4(x^2)
    # So P(x)^4 = sum Q4[j] x^{2j}
    # int P(x)^4 exp(-2x^2) dx = sum Q4[j] * int x^{2j} exp(-2x^2) dx
    from scipy.special import gamma
    total = 0.0
    for j in range(len(Q4)):
        # int x^{2j} exp(-2x^2) dx = 2^{-j-0.5} Gamma(j+0.5)
        val = (2.0 ** (-j - 0.5)) * gamma(j + 0.5)
        total += Q4[j] * val
    return total


def _integrate_poly_gauss2(c):
    """
    Compute integral of P(x)^2 exp(-x^2) dx over R.
    P(x) = sum c_{2j} x^{2j}
    P(x)^2 = sum d_m x^{2m} where d_m = sum_{i+j=m} c_{2i} c_{2j}
    int x^{2m} exp(-x^2) dx = 2^{-m-0.5} Gamma(m+0.5) * ... wait
    int x^{2m} exp(-x^2) dx = (2m)! / (2^m m!) * sqrt(pi) / 2 = ...
    Actually: int_{-inf}^{inf} x^{2m} exp(-x^2) dx = Gamma(m+0.5) * 2^{-m} * ... 
    Standard: int x^{2m} e^{-x^2} dx = (2m-1)!! * sqrt(pi) / 2^m
    = Gamma(m+0.5) * 2^{-m+0.5} ... let me just use: int x^{2m} e^{-x^2} dx = 2^{-m} Gamma(m+0.5) * 2 = ...
    Correct formula: int_{-inf}^{inf} x^{2m} e^{-x^2} dx = (2m)! / (2^m m!) * sqrt(pi) / 2
    = (2m-1)!! * sqrt(pi) / 2^m
    And (2m-1)!! = 2^m m! / sqrt(pi) * ... 
    Let me use: int x^{2m} e^{-x^2} dx = Gamma(m+0.5) * 2^{-m} * 2 / 2 ... 
    Just: int_0^inf x^{2m} e^{-x^2} dx = (1/2) Gamma(m+0.5)
    So int_{-inf}^{inf} = Gamma(m+0.5)
    Wait no: int_0^inf x^{2m} e^{-x^2} dx, let u=x^2, x=sqrt(u), dx = du/(2 sqrt(u))
    = int_0^inf u^m e^{-u} du / (2 u^{1/2}) = (1/2) int_0^inf u^{m-1/2} e^{-u} du = (1/2) Gamma(m+1/2)
    So full integral = Gamma(m+0.5)
    """
    from scipy.special import gamma
    # P(x)^2 coefficients
    # P has terms c[0], c[2], c[4], ..., c[12]
    # Let P_even = [c[0], c[2], c[4], c[6], c[8], c[10], c[12]]
    P_even = np.array([c[2*j] for j in range(7)])
    P2 = np.convolve(P_even, P_even)
    total = 0.0
    for m in range(len(P2)):
        total += P2[m] * gamma(m + 0.5)
    return total


def _bound(coeffs):
    """
    Compute C = sqrt(int |g|^4) * sqrt(int |g_hat|^4) / int |g|^2
    where g(x) = exp(-x^2/2) P(x)
    int |g|^4 = int P(x)^4 exp(-2x^2) dx
    int |g|^2 = int P(x)^2 exp(-x^2) dx
    For Fourier transform: hat{g}(xi) = (1/sqrt(2pi)) int g(x) e^{-ixi} dx
    = (1/sqrt(2pi)) int P(x) e^{-x^2/2} e^{-ixi} dx
    The FT of e^{-x^2/2} H_n(x) is (-i)^n e^{-xi^2/2} H_n(xi) (unitary convention)
    So hat{g}(xi) = (1/sqrt(2pi)) sum a_k (-i)^{4k} e^{-xi^2/2} H_{4k}(xi)
    Since (-i)^4 = 1, all signs are +1.
    So hat{g}(xi) = (1/sqrt(2pi)) e^{-xi^2/2} P(xi)
    Thus |hat{g}(xi)|^4 = (1/(2pi)^2) P(xi)^4 e^{-2xi^2}
    int |hat{g}|^4 = (1/(2pi)^2) int P(x)^4 e^{-2x^2} dx = (1/(2pi)^2) * int |g|^4
    
    Wait, that means int |hat{g}|^4 = int |g|^4 / (2pi)^2
    
    So C = sqrt(int|g|^4) * sqrt(int|g|^4/(2pi)^2) / int|g|^2
    = (int|g|^4)/(2pi) / int|g|^2
    = (int|g|^4) / (2pi * int|g|^2)
    """
    c = _poly_coeffs(coeffs)
    int_g4 = _integrate_poly_gauss4(c)
    int_g2 = _integrate_poly_gauss2(c)
    if int_g2 <= 0 or int_g4 <= 0:
        return 1e18
    # int |hat{g}|^4 = int_g4 / (2*pi)^2
    int_G4 = int_g4 / (2.0 * np.pi) ** 2
    C = np.sqrt(int_g4) * np.sqrt(int_G4) / int_g2
    return C


def construct():
    # Global optimization using differential evolution, then local refinement
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    def _bound_wrapper(x):
        try:
            return _bound(x)
        except Exception:
            return 1e18
    
    # Differential evolution for global search
    de_result = differential_evolution(
        _bound_wrapper,
        bounds=bounds,
        seed=42,
        maxiter=200,
        tol=1e-10,
        polish=False,
        popsize=15,
    )
    
    best = de_result.x
    best_val = _bound(best)
    
    # Local refinement with Nelder-Mead
    try:
        res = minimize(
            _bound,
            best,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
        )
        val = _bound(res.x)
        if val < best_val:
            best_val = val
            best = res.x
    except Exception:
        pass
    
    # Further refinement with Powell
    try:
        res2 = minimize(
            _bound,
            best,
            method="Powell",
            options={"maxiter": 3000, "ftol": 1e-14},
        )
        val = _bound(res2.x)
        if val < best_val:
            best_val = val
            best = res2.x
    except Exception:
        pass
    
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
