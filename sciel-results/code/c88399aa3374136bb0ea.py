"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    # Range [-12, 12] is sufficient for H12*exp(-x^2/2)
    x = np.linspace(-12, 12, 10000)
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
    
    # Precompute Gaussian
    gauss = np.exp(-0.5 * x**2)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
            
            # Compute norms using trapezoidal rule
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4_int = np.trapz(np.abs(g)**4, x)
            if norm4_int < 1e-12:
                return 1e10
            norm4_val = norm4_int ** 0.25
            
            # Fourier Transform using FFT
            n = len(g)
            nfft = 1
            while nfft < 2 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            
            # Scaling for Parseval's theorem in discrete context
            # \int |g|^2 \approx \sum |G|^2 * (2\pi/N)
            norm2_hat_sq = np.sum(np.abs(G)**2) * (2 * np.pi / nfft)
            norm4_hat_int = np.sum(np.abs(G)**4) * (2 * np.pi / nfft)
            
            if norm2_hat_sq < 1e-12 or norm4_hat_int < 1e-12:
                return 1e10

            norm2_val = np.sqrt(norm2_sq)
            norm2_hat_val = np.sqrt(norm2_hat_sq)
            norm4_hat_val = norm4_hat_int ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (norm2_val * norm2_hat_val)
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            # Wider range for log-magnitude to explore more
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -6 to 5
    # s_i: sign parameter, range from -1 to 1
    bounds = [(-6, 5), (-6, 5), (-6, 5), (-1, 1), (-1, 1), (-1, 1)]
    
    best_ratio = 1e10
    best_params = None
    
    # Run differential evolution with robust settings
    try:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=42, maxiter=200, popsize=25,
            tol=1e-9, polish=False,
            init='sobol',
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_params = result.x
    except Exception:
        pass
        
    # Second attempt if first was poor
    if best_ratio > 1.5:
        try:
            result2 = differential_evolution(
                objective_from_params, bounds, 
                seed=123, maxiter=150, popsize=20,
                tol=1e-9, polish=False,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if result2.fun < best_ratio:
                best_ratio = result2.fun
                best_params = result2.x
        except Exception:
            pass

    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = 1.0 if s0 > 0 else -1.0
        sign4 = 1.0 if s4 > 0 else -1.0
        sign8 = 1.0 if s8 > 0 else -1.0
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        try:
            local_result = minimize(
                local_obj, 
                [c0, c4, c8],
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if local_result.fun < best_ratio:
                c0, c4, c8 = local_result.x
                best_ratio = local_result.fun
        except Exception:
            pass
        
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
