"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial
import time


def _hermite_moment(n, k):
    """Integral of x^k * H_n(x) * exp(-x^2) dx over R."""
    if k < 0 or n < 0:
        return 0.0
    if (k + n) % 2 != 0:
        return 0.0
    m = (k + n) // 2
    # I(n,k) = 2^m * m! * (n choose 2m-n) if 2m >= n and 2m-n even
    # Actually: ∫ x^k H_n e^{-x^2} dx = 2^m m! * C(n, 2m-n) * (-1)^{2m-n} ... 
    # Let me use the standard formula more carefully.
    # ∫ x^k H_n(x) e^{-x^2} dx = 2^{(k+n)/2} * ((k+n)/2)! * (n! / ((n-k)! 2^{k/2} (k/2)!)) if k <= n and (k+n) even
    # Actually the cleanest: use the generating function or orthogonality.
    # ∫ H_n H_m e^{-x^2} dx = sqrt(pi) 2^n n! delta_{nm}
    # x^k can be expressed as sum of Hermite polynomials.
    # For now, use numerical quadrature for safety.
    pass


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) for given coefficients."""
    # Coefficients for H_n(0) to determine c12 such that g(0)=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Use a smaller grid for speed. exp(-x^2/2) is negligible for |x| > 5.
    N = 5000
    x = np.linspace(-6, 6, N)
    dx = x[1] - x[0]
    
    # Explicit Hermite polynomials for speed
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16*x2*x2 - 48*x2 + 12
    H8 = 256*x2*x2*x2*x2 - 4480*x2*x2*x2 + 26880*x2*x2 - 6720*x2 + 1680
    # H12(x) = 4096x^12 - 129024x^10 + 1761280x^8 - 11942400x^6 + 37158400x^4 - 54432000x^2 + 47900160
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096*x12 - 129024*x10 + 1761280*x8 - 11942400*x6 + 37158400*x4 - 54432000*x2 + 47900160
    
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x**2 / 2.0)
    
    # A(f) = sup |x * g(x)|
    Af = np.max(np.abs(x * g_vals))
    
    # Fourier transform using FFT
    from scipy.fft import fft, fftfreq
    G = fft(g_vals) * dx
    freqs = fftfreq(N, d=dx)
    
    AF = np.max(np.abs(freqs * G))
    
    return Af * AF


def construct():
    """Find optimal coefficients by numerical optimization."""
    import time
    from scipy.optimize import minimize
    
    start_time = time.time()
    deadline = 180.0  # Safety margin before 210s limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    def objective(coeffs):
        try:
            return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Use a limited set of start points and iterations to ensure speed
    start_points = [
        x0,
        np.array([-1.1, 0.04, 0.001]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
