import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time

def hermite(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        # Use recurrence: H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
        h0 = np.ones_like(x)
        h1 = 2 * x
        for k in range(2, n + 1):
            h2 = 2 * x * h1 - 2 * (k - 1) * h0
            h0 = h1
            h1 = h2
        return h1

def get_coefficients(a0, a4, a8):
    """Get all four coefficients given a0, a4, a8, with a12 derived from g(0)=0."""
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0  # H_4(0) = 12
    H8_0 = 1680.0  # H_8(0) = 1680
    H12_0 = 4623360.0  # H_12(0) = 4623360
    
    a12 = -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0
    return a0, a4, a8, a12

def compute_upper_bound(a0, a4, a8):
    """Compute the upper bound on C4 = A(f) * A(Fourier(f))."""
    a0, a4, a8, a12 = get_coefficients(a0, a4, a8)
    
    def f_func(x):
        h0 = hermite(0, x)
        h4 = hermite(4, x)
        h8 = hermite(8, x)
        h12 = hermite(12, x)
        g = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
        return g * np.exp(-x**2 / 2)
    
    def f_hat_func(x):
        # Fourier transform: F[H_n(x) e^{-x^2/2}] = i^n H_n(x) e^{-x^2/2} (up to normalization)
        # Using the convention where the i^n factor matters for the sign/phase
        h0 = hermite(0, x)
        h4 = hermite(4, x)
        h8 = hermite(8, x)
        h12 = hermite(12, x)
        # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1 (all real since n is multiple of 4)
        g_hat = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
        return g_hat * np.exp(-x**2 / 2)
    
    # Compute L1 norms numerically
    A_f, _ = quad(lambda x: abs(f_func(x)), -10, 10, limit=200)
    A_fhat, _ = quad(lambda x: abs(f_hat_func(x)), -10, 10, limit=200)
    
    return A_f * A_fhat

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180 second deadline with safety margin
    
    # Seed value
    seed_coeffs = np.array([-113/100, 1/25, 1/3240])
    seed_value = compute_upper_bound(*seed_coeffs)
    
    best_value = seed_value
    best_coeffs = seed_coeffs.copy()
    
    # Try a grid of values around the seed
    # First, do a coarse search
    t = time.time()
    if t < deadline - 60:
        # Try perturbations around seed
        for scale in [0.5, 0.7, 1.3, 2.0, 3.0]:
            for i in range(-5, 6):
                for j in range(-5, 6):
                    if time.time() > deadline - 60:
                        break
                    a0 = seed_coeffs[0] * scale + i * 0.01
                    a4 = seed_coeffs[1] * scale + j * 0.01
                    a8 = seed_coeffs[2] * scale + i * 0.001
                    try:
                        val = compute_upper_bound(a0, a4, a8)
                        if val < best_value:
                            best_value = val
                            best_coeffs = np.array([a0, a4, a8])
                    except:
                        pass
    
    # Local optimization
    t = time.time()
    if t < deadline - 30:
        try:
            res = minimize(
                lambda x: compute_upper_bound(x[0], x[1], x[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x
        except:
            pass
    
    return best_coeffs

# EVOLVE-BLOCK-END
