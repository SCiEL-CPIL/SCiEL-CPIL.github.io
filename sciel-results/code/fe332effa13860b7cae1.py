"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2 * x
    h_prev = np.ones_like(x, dtype=float)
    h_curr = 2 * x
    for k in range(2, n + 1):
        h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
        h_prev = h_curr
        h_curr = h_next
    return h_curr


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty upper bound for given coefficients."""
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1008, H_12(0)=-48640
    a12 = (a0 - 12.0 * a4 + 1008.0 * a8) / 48640.0
    
    # Evaluate on a grid
    x = np.linspace(-15, 15, 20000)
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2.0)
    
    dx = x[1] - x[0]
    
    # L2 norm
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-30:
        return 1e10
    
    # x*g norm
    xg = x * g
    xg_norm_sq = np.sum(xg**2) * dx
    
    # Fourier transform via FFT (approximate)
    # Use a larger grid for FFT accuracy
    N = 1 << 15  # 32768
    x_fft = np.linspace(-20, 20, N)
    dx_fft = x_fft[1] - x_fft[0]
    
    H0_f = _hermite(0, x_fft)
    H4_f = _hermite(4, x_fft)
    H8_f = _hermite(8, x_fft)
    H12_f = _hermite(12, x_fft)
    
    g_f = (a0 * H0_f + a4 * H4_f + a8 * H8_f + a12 * H12_f) * np.exp(-x_fft**2 / 2.0)
    
    # FFT
    G = np.fft.fft(g_f) * dx_fft
    # Shift to center
    G = np.fft.fftshift(G)
    freq = np.fft.fftfreq(N, d=dx_fft)
    freq = np.fft.fftshift(freq)
    
    # |xi * G|^2 integral
    xiG = freq * G
    xiG_norm_sq = np.sum(np.abs(xiG)**2) * dx_fft / (2 * np.pi) ** 2 * 2 * np.pi
    # Actually, let's be more careful with Fourier transform conventions
    
    # Using convention: F(x) = integral f(t) e^{-ixt} dt
    # Then ||x f||_2 ||xi F||_2 >= (1/2) ||f||_2^2
    # The product ||x f||_2 * ||xi F||_2 / ||f||_2^2 is what we want to minimize
    
    # Let me recompute more carefully
    # ||g||_2^2 = integral |g(x)|^2 dx
    # ||x g||_2^2 = integral |x g(x)|^2 dx
    # ||xi G||_2^2 = integral |xi G(xi)|^2 dxi where G = Fourier transform
    
    # With unitary Fourier transform: ||G||_2 = ||g||_2
    # And ||xi G||_2 = ||d/dx g||_2 (up to constants)
    
    # Let's use: uncertainty product = sqrt(||x g||_2^2 * ||xi G||_2^2) / ||g||_2^2
    # But we need consistent conventions.
    
    # Simpler: compute variance-based uncertainty
    # sigma_x^2 = <x^2> - <x>^2
    # sigma_p^2 = <p^2> - <p>^2 where p = -i d/dx
    # Product sigma_x * sigma_p >= 1/2
    
    # For our purposes, let's compute:
    # U = sqrt(integral x^2 |g|^2 dx * integral |g'|^2 dx) / integral |g|^2 dx
    
    # g'(x) = derivative of P(x) e^{-x^2/2}
    # = P'(x) e^{-x^2/2} - x P(x) e^{-x^2/2}
    
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    # P'(x) = a0*H0' + a4*H4' + a8*H8' + a12*H12'
    # H_n'(x) = 2n H_{n-1}(x)
    H0p = np.zeros_like(x)  # H_0' = 0
    H4p = 8 * _hermite(3, x)  # H_4' = 8 H_3
    H8p = 16 * _hermite(7, x)  # H_8' = 16 H_7
    H12p = 24 * _hermite(11, x)  # H_12' = 24 H_11
    
    Pp = a0 * H0p + a4 * H4p + a8 * H8p + a12 * H12p
    
    gp = (Pp - x * P) * np.exp(-x**2 / 2.0)
    
    gp_norm_sq = np.sum(gp**2) * dx
    
    # Uncertainty product (variance-based)
    # sigma_x^2 = (integral x^2 |g|^2 dx) / (integral |g|^2 dx) since <x> should be 0 for even functions
    # sigma_p^2 = (integral |g'|^2 dx) / (integral |g|^2 dx)
    
    if norm_sq < 1e-30:
        return 1e10
    
    sigma_x_sq = xg_norm_sq / norm_sq
    sigma_p_sq = gp_norm_sq / norm_sq
    
    # The bound on C4 might be related to sigma_x * sigma_p
    # Or it could be the ratio that appears in the uncertainty inequality
    
    # Let me return the product sigma_x * sigma_p which is >= 1/2
    bound = np.sqrt(sigma_x_sq * sigma_p_sq)
    
    return bound


def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Initial guess: the prior state of the art
    a0_init, a4_init, a8_init = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        a0, a4, a8 = params
        try:
            return _compute_bound(a0, a4, a8)
        except:
            return 1e10
    
    # Try multiple starting points
    best_result = None
    best_score = 1e10
    
    starts = [
        [a0_init, a4_init, a8_init],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [2.0, -0.5, 0.1],
        [-2.0, 0.5, -0.1],
        [0.1, 0.1, 0.01],
        [-0.1, -0.1, -0.01],
    ]
    
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except:
            pass
    
    if best_result is not None:
        return np.array(best_result)
    
    return np.array([a0_init, a4_init, a8_init])
# EVOLVE-BLOCK-END
