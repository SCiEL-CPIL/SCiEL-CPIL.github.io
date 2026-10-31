"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    x = np.linspace(-20, 20, 10000)
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
            
            # Compute norms using trapezoidal rule for better accuracy
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4 = np.trapz(np.abs(g)**4, x)
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT with proper scaling
            # Pad to next power of 2 for efficiency
            n = len(g)
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            freq = np.fft.fftfreq(nfft, d=dx)
            
            # Compute integrals using trapezoidal rule in frequency domain
            domega = 2 * np.pi / (nfft * dx)
            norm2_hat_sq = np.sum(np.abs(G)**2) * domega
            norm4_hat = np.sum(np.abs(G)**4) * domega
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use smooth sign: sign = tanh(s) to avoid discontinuities
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            
            # Avoid zero coefficients
            min_mag = 0.01
            c0 = max(min_mag, np.exp(m0)) * sign0 if abs(sign0) > 0.01 else min_mag
            c4 = max(min_mag, np.exp(m4)) * sign4 if abs(sign4) > 0.01 else min_mag
            c8 = max(min_mag, np.exp(m8)) * sign8 if abs(sign8) > 0.01 else min_mag
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -3 to 2 (coefficients from ~0.05 to ~7.4)
    # s_i: sign parameter, range from -3 to 3 (tanh maps to ~[-0.995, 0.995])
    bounds = [(-3, 2), (-3, 2), (-3, 2), (-3, 3), (-3, 3), (-3, 3)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    best_coeffs = None
    
    for seed in [42, 123, 456, 789, 1024]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=25,
            tol=1e-10, polish=False,
            init='sobol',
            mutation=(0.6, 1.2)
        )
        
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_params = result.x
            m0, m4, m8, s0, s4, s8 = best_params
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            min_mag = 0.01
            c0 = max(min_mag, np.exp(m0)) * sign0 if abs(sign0) > 0.01 else min_mag
            c4 = max(min_mag, np.exp(m4)) * sign4 if abs(sign4) > 0.01 else min_mag
            c8 = max(min_mag, np.exp(m8)) * sign8 if abs(sign8) > 0.01 else min_mag
            best_coeffs = np.array([c0, c4, c8])
    
    if best_coeffs is not None:
        c0, c4, c8 = best_coeffs
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
        
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
