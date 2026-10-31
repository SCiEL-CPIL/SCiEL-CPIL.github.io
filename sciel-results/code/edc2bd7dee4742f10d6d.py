"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Use a finer grid for higher accuracy in numerical integration
    # 16000 points over [-15, 15] provides dx ~ 0.0019, very accurate for Gaussians
    x = np.linspace(-15, 15, 16000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            # Compute norms using trapezoidal rule
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4 = np.trapz(np.abs(g)**4, x)
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT with proper scaling
            # Pad to next power of 2, ensure sufficient size to avoid aliasing
            n = len(g)
            nfft = 1
            # Use 4x padding to be safe with spectral tails
            while nfft < 4 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            
            # The integral of |G|^2 is sum(|G|^2) * (2*pi/nfft) ?
            # Parseval: sum |g|^2 dx = sum |G|^2 * (2*pi / nfft) ?
            # Standard FFT: G_k = sum g_j exp(-i 2 pi j k / N)
            # Inverse: g_j = 1/N sum G_k exp(i 2 pi j k / N)
            # Continuous FT: F(omega) = int f(t) e^{-i omega t} dt
            # Discrete approx: F_k ~ dx * sum g_j e^{-i 2 pi j k / N} = dx * G_k
            # So F_k = dx * G_k
            # Integral |F|^2 domega = sum |F_k|^2 * domega
            # domega = 2*pi / T_total = 2*pi / (N * dx)
            # So Integral = sum |dx * G_k|^2 * (2*pi / (N * dx))
            # = dx^2 * sum |G_k|^2 * 2*pi / (N * dx)
            # = dx * 2*pi / N * sum |G_k|^2
            # Note: N here is nfft.
            
            norm2_hat_sq = np.sum(np.abs(G)**2) * (2 * np.pi * dx / nfft)
            norm4_hat = np.sum(np.abs(G)**4) * (2 * np.pi * dx / nfft)
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use smooth sign transition to help DE? No, discrete sign is better for DE.
            # But we can use a continuous mapping if we want.
            # Let's stick to discrete sign for clarity and robustness in DE.
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # Wider bounds to explore more of the landscape
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-1, 1), (-1, 1), (-1, 1)]
    
    # Run differential evolution
    best_ratio = 1e10
    best_params = None
    
    # Use multiple seeds and slightly larger population for better exploration
    for seed in [42, 123, 456, 789]:
        try:
            result = differential_evolution(
                objective_from_params, bounds, 
                seed=seed, maxiter=200, popsize=25,
                tol=1e-10, polish=False,
                init='sobol',
                mutation=(0.4, 1.0),
                recombination=0.8
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_params = result.x
        except Exception:
            continue
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = 1.0 if s0 > 0 else -1.0
        sign4 = 1.0 if s4 > 0 else -1.0
        sign8 = 1.0 if s8 > 0 else -1.0
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        # Try multiple local optimization methods and starting points
        best_local_ratio = best_ratio
        best_local_coeffs = np.array([c0, c4, c8])
        
        # 1. Nelder-Mead
        try:
            res_nm = minimize(
                local_obj, 
                [c0, c4, c8],
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < best_local_ratio:
                best_local_ratio = res_nm.fun
                best_local_coeffs = res_nm.x
        except Exception:
            pass
        
        # 2. Powell (derivative-free, often better for smooth non-convex)
        try:
            res_powell = minimize(
                local_obj, 
                best_local_coeffs,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            if res_powell.fun < best_local_ratio:
                best_local_ratio = res_powell.fun
                best_local_coeffs = res_powell.x
        except Exception:
            pass
            
        # 3. One more Nelder-Mead from the Powell result to ensure convergence
        try:
            res_nm2 = minimize(
                local_obj, 
                best_local_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm2.fun < best_local_ratio:
                best_local_ratio = res_nm2.fun
                best_local_coeffs = res_nm2.x
        except Exception:
            pass

        c0, c4, c8 = best_local_coeffs
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
