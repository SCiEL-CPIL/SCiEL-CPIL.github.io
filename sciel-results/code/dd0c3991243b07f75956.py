import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    # Hermite polynomial values at 0 (physicists' convention)
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 560.0
    H12_0 = 425280.0
    
    def H(n, x):
        """Physicists' Hermite polynomial."""
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
            return 256 * x**8 - 3072 * x**6 + 11520 * x**4 - 13440 * x**2 + 1680
        elif n == 12:
            return 4096 * x**12 - 73728 * x**10 + 483840 * x**8 - 1445760 * x**6 + 1814400 * x**4 - 806400 * x**2 + 40840
        else:
            raise ValueError(f"H({n}) not implemented")
    
    def get_a12(a0, a4, a8):
        """Compute a12 from P(0) = 0 constraint."""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def compute_bound(coeffs):
        """Compute the upper bound A(f) * A(F) for given coefficients."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        # Use a fine grid for numerical integration
        x = np.linspace(-15, 15, 20000)
        dx = x[1] - x[0]
        
        # Build P(x)
        P = a0 * H(0, x) + a4 * H(4, x) + a8 * H(8, x) + a12 * H(12, x)
        
        # f(x) = exp(-x^2/4) * P(x)
        f = np.exp(-x**2 / 4) * P
        f2 = f**2
        
        # A(f) = ∫x²|f|² dx / ∫|f|² dx
        norm_f = np.sum(f2) * dx
        if norm_f < 1e-15:
            return 1e10
        var_f = np.sum(x**2 * f2) * dx / norm_f
        
        # Compute Fourier transform numerically using FFT
        # Use a sufficiently large and well-spaced grid
        nfft = 2**15
        xfft = np.linspace(-30, 30, nfft, endpoint=False)
        dxfft = xfft[1] - xfft[0]
        
        P_fft_grid = a0 * H(0, xfft) + a4 * H(4, xfft) + a8 * H(8, xfft) + a12 * H(12, xfft)
        f_fft = np.exp(-xfft**2 / 4) * P_fft_grid
        
        # FFT to get F(ξ)
        F_fft = np.fft.fftshift(np.fft.fft(f_fft * dxfft))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, dxfft))
        
        F2 = np.abs(F_fft)**2
        norm_F = np.sum(F2) * (freq[1] - freq[0])
        if norm_F < 1e-15:
            return 1e10
        var_F = np.sum(freq**2 * F2) * (freq[1] - freq[0]) / norm_F
        
        bound = var_f * var_F
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        return bound
    
    # Seed coefficients (prior state of the art)
    seed_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    best_val = compute_bound(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Try careful local optimization from seed
    if time.time() < deadline - 60:
        try:
            res = minimize(
                compute_bound,
                seed_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Try a few small perturbations around seed
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            perturbation = rng.normal(0, 0.01, 3)
            x_try = seed_coeffs + perturbation
            try:
                val = compute_bound(x_try)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_coeffs = x_try.copy()
            except Exception:
                continue
    
    return best_coeffs
