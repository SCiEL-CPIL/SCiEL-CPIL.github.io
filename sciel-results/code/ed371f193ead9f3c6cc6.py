"""Optimized search for Hermite coefficients minimizing the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


# Precompute Hermite polynomial coefficients for fast evaluation
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 75600x^2 + 10395

def _hermite_coeffs():
    """Return coefficients for H0, H4, H8, H12 in standard (probabilists') form."""
    return np.array([1.0, 1.0, 1.0, 1.0])


def _eval_hermite_fast(x, c0, c4, c8, c12):
    """Evaluate the Hermite combination at x using precomputed powers."""
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x8 * x4
    
    h0 = 1.0
    h4 = x4 - 6.0 * x2 + 3.0
    h8 = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
    h12 = x12 - 60.0 * x10 + 1260.0 * x8 - 12600.0 * x6 + 56700.0 * x4 - 75600.0 * x2 + 10395.0
    
    return c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12


def _compute_bound(c0, c4, c8):
    """Compute the C4 upper bound given c0, c4, c8 with c12 fixed by g(0)=0."""
    # g(0) = c0 + c4*3 + c8*105 + c12*10395 = 0
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
    
    # Use a finer grid for better accuracy
    x = np.linspace(-12, 12, 16384)
    dx = x[1] - x[0]
    
    # f(x) = g(x) * exp(-x^2/2)
    g_vals = _eval_hermite_fast(x, c0, c4, c8, c12)
    gauss = np.exp(-x**2 / 2.0)
    f_vals = g_vals * gauss
    
    # A(f) = ||f||_inf / ||f||_1
    sup_f = np.max(np.abs(f_vals))
    int_f = np.sum(np.abs(f_vals)) * dx
    
    if int_f < 1e-15:
        return 1e10  # invalid
    
    # Compute Fourier transform using FFT
    # f_hat(k) = integral f(x) exp(-ikx) dx
    # Use FFT with proper scaling
    n = len(f_vals)
    n_fft = 1 << (n - 1).bit_length()  # next power of 2
    n_fft = max(n_fft, 32768)
    
    # Place f in center of FFT array
    f_padded = np.zeros(n_fft, dtype=complex)
    mid = n_fft // 2
    start = mid - n // 2
    f_padded[start:start + n] = f_vals
    
    # FFT with proper scaling
    # f_hat[k] = dx * sum f[x] * exp(-2*pi*i*k*x/L)
    # where L = n*dx is the total length
    f_hat = np.fft.fftshift(np.fft.fft(f_padded)) * dx
    
    # Frequency spacing: dk = 1/L = 1/(n*dx)
    dk = 1.0 / (n * dx)
    
    sup_fh = np.max(np.abs(f_hat))
    int_fh = np.sum(np.abs(f_hat)) * dk
    
    if int_fh < 1e-15:
        return 1e10  # invalid
    
    A_f = sup_f / int_f
    A_fh = sup_fh / int_fh
    
    return A_f * A_fh


def _objective_and_gradient(params):
    """Compute objective and numerical gradient for L-BFGS-B."""
    c0, c4, c8 = params
    val = _compute_bound(c0, c4, c8)
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros(3)
    grad[0] = (_compute_bound(c0 + eps, c4, c8) - _compute_bound(c0 - eps, c4, c8)) / (2 * eps)
    grad[1] = (_compute_bound(c0, c4 + eps, c8) - _compute_bound(c0, c4 - eps, c8)) / (2 * eps)
    grad[2] = (_compute_bound(c0, c4, c8 + eps) - _compute_bound(c0, c4, c8 - eps)) / (2 * eps)
    
    return val, grad


def construct():
    """Find optimal Hermite coefficients minimizing the C4 upper bound."""
    t_start = time.time()
    deadline = t_start + 175.0  # Safety margin before 210s kill
    
    def time_left():
        return deadline - time.time()
    
    # Initial guess from prior work
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = _compute_bound(*best_coeffs)
    
    # Phase 1: Coarse grid search to find good starting points
    if time_left() > 60.0:
        c0_range = np.linspace(-2.0, 0.5, 12)
        c4_range = np.linspace(-0.5, 0.5, 10)
        c8_range = np.linspace(-0.1, 0.1, 8)
        
        for c0_init in c0_range:
            for c4_init in c4_range:
                for c8_init in c8_range:
                    if time_left() < 45.0:
                        break
                    try:
                        val = _compute_bound(c0_init, c4_init, c8_init)
                        if val < best_bound:
                            best_bound = val
                            best_coeffs = np.array([c0_init, c4_init, c8_init])
                    except Exception:
                        pass
                if time_left() < 45.0:
                    break
            if time_left() < 45.0:
                break
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starting points
    if time_left() > 30.0:
        # Try refinement from best point and a few nearby points
        start_points = [best_coeffs.copy()]
        
        # Add some perturbed versions
        for scale in [0.1, 0.05]:
            for i in range(3):
                if time_left() < 25.0:
                    break
                perturbation = np.array([
                    (np.random.rand() - 0.5) * scale,
                    (np.random.rand() - 0.5) * scale,
                    (np.random.rand() - 0.5) * scale
                ])
                start_points.append(best_coeffs + perturbation)
        
        for x0 in start_points:
            if time_left() < 20.0:
                break
            try:
                result = minimize(
                    _compute_bound, x0, method='L-BFGS-B',
                    jac=lambda x: np.array([
                        (_compute_bound(x[0] + 1e-7, x[1], x[2]) - 
                         _compute_bound(x[0] - 1e-7, x[1], x[2])) / 2e-7,
                        (_compute_bound(x[0], x[1] + 1e-7, x[2]) - 
                         _compute_bound(x[0], x[1] - 1e-7, x[2])) / 2e-7,
                        (_compute_bound(x[0], x[1], x[2] + 1e-7) - 
                         _compute_bound(x[0], x[1], x[2] - 1e-7)) / 2e-7
                    ]),
                    options={'gtol': 1e-12, 'maxiter': 300, 'maxls': 50}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_coeffs = result.x
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if time_left() > 10.0:
        try:
            result = minimize(
                _compute_bound, best_coeffs, method='L-BFGS-B',
                jac=lambda x: np.array([
                    (_compute_bound(x[0] + 1e-8, x[1], x[2]) - 
                     _compute_bound(x[0] - 1e-8, x[1], x[2])) / 2e-8,
                    (_compute_bound(x[0], x[1] + 1e-8, x[2]) - 
                     _compute_bound(x[0], x[1] - 1e-8, x[2])) / 2e-8,
                    (_compute_bound(x[0], x[1], x[2] + 1e-8) - 
                     _compute_bound(x[0], x[1], x[2] - 1e-8)) / 2e-8
                ]),
                options={'gtol': 1e-14, 'maxiter': 500, 'maxls': 100}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2]])
# EVOLVE-BLOCK-END
