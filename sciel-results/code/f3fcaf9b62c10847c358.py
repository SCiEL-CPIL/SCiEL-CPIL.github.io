"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

_DEADLINE = 180  # safety margin before 210s kill

def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        # H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 12:
        # H_12(x) = 4096x^12 - 147456x^10 + 2580480x^8 - 20321280x^6 + 81729600x^4 - 172972800x^2 + 103219200
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

def _compute_norm_sq(coeffs, x):
    """Compute ||g||^2 = ∫ g(x)^2 dx where g(x) = (sum a_i H_{d_i}(x)) * exp(-x^2/2).
    
    coeffs = [a0, a4, a8, a12] for degrees [0, 4, 8, 12]
    """
    a0, a4, a8, a12 = coeffs
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # |g|^2 = (a0*H0 + a4*H4 + a8*H8 + a12*H12)^2 * exp(-x^2)
    
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    g_poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    integrand = g_poly**2 * np.exp(-x**2)
    
    return np.trapz(integrand, x)

def _compute_fourier_norm_sq(coeffs, x):
    """Compute ||F(g)||^2 = ∫ |F(g)(ξ)|^2 dξ.
    
    For g(x) = P(x)*exp(-x^2/2), the Fourier transform is related to derivatives of P.
    Using Parseval: ||F(g)||^2 = ||g||^2 (if Fourier transform is unitary).
    Actually for the standard Fourier transform F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx,
    Parseval gives ||F(g)||^2 = ||g||^2.
    
    But we need A(f) = ||f||_1 or similar. Let me reconsider.
    
    The uncertainty inequality involves A(f) * A(F(f)) where A(h) = ∫|h(x)|dx or similar.
    For the Hermite-Gaussian case, we can compute these norms.
    """
    # For unitary Fourier transform, ||F(g)||_2 = ||g||_2
    # But A(h) might be ||h||_1 or ||h||_∞ or another norm
    # Let's use the L1 norm for now
    a0, a4, a8, a12 = coeffs
    
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    g_poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = g_poly * np.exp(-x**2/2)
    
    return np.trapz(np.abs(g), x)

def _c4_bound_from_coeffs(coeffs):
    """Compute the C4 upper bound for given coefficients [a0, a4, a8, a12].
    
    The bound is on A(f) * A(F(f)). For the test function g(x) = P(x)*exp(-x^2/2),
    we compute the relevant norms.
    """
    if time.time() > _DEADLINE - 30:
        return float('inf')
    
    x = np.linspace(-20, 20, 10000)
    
    # Compute ||g||_1 and ||F(g)||_1 or appropriate norms
    # For the uncertainty principle, typically A(f) = ∫|f(x)|dx
    # and the Fourier transform of P(x)*exp(-x^2/2) can be computed
    
    a0, a4, a8, a12 = coeffs
    
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    g_poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = g_poly * np.exp(-x**2/2)
    
    # A(g) = ∫|g(x)|dx
    A_g = np.trapz(np.abs(g), x)
    
    # For the Fourier transform, use FFT or analytic
    # F(g)(ξ) = ∫ g(x) e^{-2πixξ} dx
    # For g(x) = P(x)exp(-x^2/2), this is related to the characteristic function
    
    # Use numerical FFT for simplicity
    n = len(x)
    dx = x[1] - x[0]
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    
    # A(F(g)) = ∫|F(g)(ξ)|dξ
    A_Fg = np.trapz(np.abs(G), freqs)
    
    return A_g * A_Fg

def construct():
    start_time = time.time()
    
    # The constraint g(0) = 0 fixes a12 in terms of a0, a4, a8
    # H0(0) = 1, H4(0) = 12, H8(0) = 40320, H12(0) = 103219200
    # So: a0 + 12*a4 + 40320*a8 + 103219200*a12 = 0
    # => a12 = -(a0 + 12*a4 + 40320*a8) / 103219200
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 40320.0
    H12_0 = 103219200.0
    
    def objective(params):
        if time.time() > _DEADLINE - 30:
            return float('inf')
        a0, a4, a8 = params
        a12 = -(a0 + H4_0*a4 + H8_0*a8) / H12_0
        coeffs = [a0, a4, a8, a12]
        return _c4_bound_from_coeffs(coeffs)
    
    # Start from the seed solution
    seed = [-113/100, 1/25, 1/3240]
    
    best_result = None
    best_score = float('inf')
    
    # Try local optimization from seed
    try:
        result = minimize(objective, seed, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    if time.time() < _DEADLINE - 60:
        for _ in range(5):
            if time.time() > _DEADLINE - 60:
                break
            try:
                random_start = np.random.uniform(-2, 2, 3)
                result = minimize(objective, random_start, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x
            except Exception:
                pass
    
    if best_result is not None:
        return np.array(best_result)
    else:
        return np.array(seed)

# EVOLVE-BLOCK-END
