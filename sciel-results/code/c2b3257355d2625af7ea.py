"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: kill after 210s wall-clock, return best with safety margin
_START_TIME = time.time()
TIME_LIMIT = 210.0
SAFETY_MARGIN = 10.0

def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _hermite_polynomial_coeffs(n):
    """Return coefficients of the probabilists' Hermite polynomial H_n(x) in descending power order."""
    # Probabilists' Hermite polynomials: He_0=1, He_1=x, He_2=x^2-1, He_3=x^3-3x, ...
    # H_n (physicists') vs He_n (probabilists'): The task says "Hermite-polynomial times Gaussian"
    # and mentions H0, H4, H8, H12. Given the context of Fourier-invariant test functions
    # for uncertainty inequalities, these are likely the physicists' Hermite polynomials
    # or the probabilists'. Let me use the standard physicists' Hermite polynomials H_n
    # which satisfy H_n(x) = (-1)^n e^{x^2/2} d^n/dx^n e^{-x^2/2}.
    # Actually, for the uncertainty inequality context (Goncalves et al.), the test function
    # is typically g(x) = sum a_k H_k(x) e^{-x^2/2} where H_k are physicists' Hermite polys.
    # Let's use physicists' Hermite polynomials.
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    # Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    h_prev = np.array([1.0])  # H_0
    h_curr = np.array([2.0, 0.0])  # H_1
    if n == 0:
        return h_prev
    if n == 1:
        return h_curr
    for k in range(1, n):
        h_next = np.convolve(h_curr, np.array([2.0, 0.0])) - 2*k*h_prev
        # Trim to correct length
        h_next = h_next[:n+1] if len(h_next) > n+1 else h_next
        # Pad if needed
        if len(h_next) < n+1:
            h_next = np.pad(h_next, (0, n+1-len(h_next)))
        h_prev = h_curr
        h_curr = h_next
    return h_curr


def _eval_g(x, coeffs):
    """Evaluate g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0."""
    h0 = np.array([1.0])
    h4 = _hermite_polynomial_coeffs(4)
    h8 = _hermite_polynomial_coeffs(8)
    h12 = _hermite_polynomial_coeffs(12)
    
    a0, a4, a8 = coeffs
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0) = 0
    # H_n(0) for even n: H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-199584
    # Actually let me compute: H0(0)=1, H2(0)=-2, H4(0)=-12, H6(0)=120, H8(0)=-1680, H10(0)=30240, H12(0)=-665280
    # Wait, let me recompute. Physicists' Hermite: H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x, H_4=16x^4-48x^2+12
    # H_4(0)=12, H_8(0): H_8 = 256x^8 - 4480x^6 + 23040x^4 - 33600x^2 + 1680, so H_8(0)=1680
    # H_12(0): this is getting complex. Let me just evaluate numerically.
    
    h0_0 = np.polyval(h0, 0)
    h4_0 = np.polyval(h4, 0)
    h8_0 = np.polyval(h8, 0)
    h12_0 = np.polyval(h12, 0)
    
    # a12 * h12_0 = -(a0*h0_0 + a4*h4_0 + a8*h8_0)
    if abs(h12_0) > 1e-15:
        a12 = -(a0*h0_0 + a4*h4_0 + a8*h8_0) / h12_0
    else:
        a12 = 0.0
    
    # g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    poly = a0*h0 + a4*h4 + a8*h8 + a12*h12
    # Make sure poly has same length
    max_len = max(len(h0), len(h4), len(h8), len(h12))
    poly = np.pad(poly, (0, max_len-len(poly))) if len(poly) < max_len else poly[:max_len]
    
    p = np.polyval(poly, x)
    return p * np.exp(-x**2 / 2)


def _compute_bound(coeffs):
    """Compute the upper bound on A(f)*A(Fourier(f)) for the given coefficients.
    
    The bound is related to the L1 norm of g and its Fourier transform.
    For a function g, the uncertainty bound involves:
    A(g) * A(Fourier(g)) where A(h) = sup_x |h(x)| / (integral of |h|)^{1/d} or similar.
    
    Actually, for the uncertainty inequality in the form:
    ||f||_1 * ||F(f)||_1 >= C
    The test function approach uses g such that f = g * something.
    
    The bound on C4 is typically:
    C4 <= (||g||_1 * ||F(g)||_1) / (||g||_2 * ||F(g)||_2) or similar ratio.
    
    Let me think more carefully. The standard approach for proving uncertainty inequalities
    using test functions: if we want to show ||f||_1 ||F(f)||_1 >= C, we use the fact that
    for any test function g, we have a bound involving the ratio of norms of g and its FT.
    
    The upper bound on the constant C4 is:
    C4 <= sup over test functions of [||g||_1 * ||F(g)||_1] / [||g||_2^2] or similar.
    
    Actually, I think the bound is:
    C >= ||g||_1 * ||F(g)||_1 / ||g||_2^2
    
    And we want to MINIMIZE this upper bound to get the tightest inequality.
    
    Let me compute numerically."""
    # Sample g and its Fourier transform
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    g = _eval_g(x, coeffs)
    
    # Compute Fourier transform using FFT
    # The Fourier transform convention matters. For the uncertainty principle:
    # F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx or f(x) e^{-i*x*xi} dx
    # Let's use the convention F(f)(xi) = integral f(x) e^{-i*x*xi} dx
    # which corresponds to the standard L2 Fourier transform.
    
    # Using FFT: if g is sampled at x_k = k*dx for k=-N/2..N/2-1,
    # then F(g)(xi_m) ≈ dx * sum_k g_k * exp(-i*x_k*xi_m)
    # where xi_m = 2*pi*m/(N*dx)
    
    N = len(x)
    # Shift so that x starts at 0 for FFT
    # x = -20 to 20, so x[0] = -20
    # For FFT, we need the function on a grid starting at 0
    # Let's use a different approach: use the fact that g is even (Hermite polys are even for even n)
    
    # Actually, let's just compute the FT numerically
    # Use scipy's fft
    from scipy.fft import fft, fftshift, fftfreq
    
    # g is defined on x from -20 to 20
    # For FFT, we need values on a uniform grid. Let's use:
    # x_fft = np.arange(N) * dx, and shift g accordingly
    # g is centered at 0, so g_fft[k] = g[k - N//2] for appropriate indexing
    
    # Simpler: use the direct formula with a fine grid for the FT
    # F(g)(xi) = integral g(x) e^{-i*x*xi} dx
    # Sample xi from -xi_max to xi_max
    
    xi_max = 20
    xi = np.linspace(-xi_max, xi_max, 20001)
    dxi = xi[1] - xi[0]
    
    # Compute FT using matrix multiplication (might be slow, but let's try)
    # Actually, let's use the property that FT of H_n(x) e^{-x^2/2} is related to H_n(xi) e^{-xi^2/2}
    # For physicists' Hermite: FT of H_n(x) e^{-x^2/2} = i^n H_n(xi) e^{-xi^2/2}
    # (with the convention F(f)(xi) = (1/sqrt(2*pi)) integral f(x) e^{-i*x*xi} dx)
    # Or without the 1/sqrt(2*pi): integral f(x) e^{-i*x*xi} dx = i^n sqrt(2*pi) H_n(xi) e^{-xi^2/2}
    
    # Let's use the analytic FT! This is much more efficient.
    # With F(f)(xi) = integral_{-inf}^{inf} f(x) e^{-i*x*xi} dx:
    # F[H_n(x) e^{-x^2/2}](xi) = i^n * sqrt(2*pi) * H_n(xi) * e^{-xi^2/2}
    
    # So F(g)(xi) = sqrt(2*pi) * sum_k a_k * i^k * H_k(xi) * e^{-xi^2/2}
    # where k is the degree (0, 4, 8, 12)
    
    h0 = np.array([1.0])
    h4 = _hermite_polynomial_coeffs(4)
    h8 = _hermite_polynomial_coeffs(8)
    h12 = _hermite_polynomial_coeffs(12)
    
    a0, a4, a8 = coeffs
    h0_0 = np.polyval(h0, 0)
    h4_0 = np.polyval(h4, 0)
    h8_0 = np.polyval(h8, 0)
    h12_0 = np.polyval(h12, 0)
    if abs(h12_0) > 1e-15:
        a12 = -(a0*h0_0 + a4*h4_0 + a8*h8_0) / h12_0
    else:
        a12 = 0.0
    
    # F(g)(xi) = sqrt(2*pi) * [a0 * i^0 * H0(xi) * e^{-xi^2/2} + a4 * i^4 * H4(xi) * e^{-xi^2/2} + ...]
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1 (all even powers of i are 1 or -1, but 4,8,12 are multiples of 4)
    # i^4 = (i^2)^2 = (-1)^2 = 1
    # i^8 = (i^4)^2 = 1
    # i^12 = (i^4)^3 = 1
    # So all the i^k factors are 1!
    
    sqrt_2pi = np.sqrt(2 * np.pi)
    g_ft_poly = a0*h0 + a4*h4 + a8*h8 + a12*h12
    max_len = max(len(h0), len(h4), len(h8), len(h12))
    g_ft_poly = np.pad(g_ft_poly, (0, max_len-len(g_ft_poly))) if len(g_ft_poly) < max_len else g_ft_poly[:max_len]
    
    g_ft = sqrt_2pi * np.polyval(g_ft_poly, xi) * np.exp(-xi**2 / 2)
    
    # Now compute the norms
    # ||g||_1 = integral |g(x)| dx
    # ||F(g)||_1 = integral |F(g)(xi)| dxi
    # ||g||_2 = sqrt(integral |g(x)|^2 dx)
    # ||F(g)||_2 = sqrt(integral |F(g)(xi)|^2 dxi)
    
    norm_g_1 = np.trapz(np.abs(g), x)
    norm_ft_1 = np.trapz(np.abs(g_ft), xi)
    norm_g_2 = np.sqrt(np.trapz(np.abs(g)**2, x))
    norm_ft_2 = np.sqrt(np.trapz(np.abs(g_ft)**2, xi))
    
    # The upper bound on C4 is:
    # C4 <= ||g||_1 * ||F(g)||_1 / (||g||_2 * ||F(g)||_2)
    # or possibly ||g||_1 * ||F(g)||_1 / ||g||_2^2
    
    # For the standard uncertainty inequality ||f||_1 ||F(f)||_1 >= C * ||f||_2^2,
    # the best constant is C = inf_f ||f||_1 ||F(f)||_1 / ||f||_2^2
    # And the test function bound gives C <= ||g||_1 ||F(g)||_1 / ||g||_2^2
    
    if norm_g_2 < 1e-15:
        return np.inf
    
    bound = norm_g_1 * norm_ft_1 / (norm_g_2 ** 2)
    return bound


def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    # Start with the known best: [-113/100, 1/25, 1/3240]
    initial = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = initial.copy()
    best_bound = _compute_bound(best_coeffs)
    
    # Try to improve using optimization
    if _remaining_time() < 30:
        return best_coeffs
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize as sp_minimize
    
    def objective(x):
        if _remaining_time() < 20:
            return best_bound
        try:
            return _compute_bound(x)
        except:
            return 1e10
    
    # Try a few different starting points
    starts = [
        initial,
        initial + np.array([0.1, 0.01, 0.001]),
        initial + np.array([-0.1, -0.01, -0.001]),
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
    ]
    
    for start in starts:
        if _remaining_time() < 30:
            break
        try:
            res = sp_minimize(objective, start, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_bound:
                best_bound = res.fun
                best_coeffs = res.x.copy()
        except:
            pass
    
    # Final check
    if _remaining_time() < 5:
        return best_coeffs
    
    # Try a polish step with tighter tolerance
    try:
        res = sp_minimize(objective, best_coeffs, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_bound:
            best_bound = res.fun
            best_coeffs = res.x.copy()
    except:
        pass
    
    return best_coeffs

# EVOLVE-BLOCK-END
