"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    # Range [-15, 15] is sufficient for exp(-0.5*x^2)
    x = np.linspace(-15, 15, 10000)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint g(0)=0
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Precompute FFT parameters
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            # Compute norms in x-domain
            # Using sum * dx for uniform grid is equivalent to trapz for large N
            g2 = g**2
            norm2_sq = np.sum(g2) * dx
            if norm2_sq < 1e-15:
                return 1e10
                
            g4 = g2**2
            norm4 = np.sum(g4) * dx
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform
            # Using FFT. G_k = sum g_j exp(-i 2pi j k / N) * dx
            G = np.fft.fft(g, n=nfft) * dx
            
            # Parseval's theorem scaling for discrete FFT
            # Integral |G|^2 dxi approx sum |G_k|^2 * dxi
            # dxi = 1 / (N_fft * dx) if using cyclic freq convention?
            # Standard: if h = fft(g), then IFT(h) = g.
            # Continuous FT: F(omega) = int g(x) e^{-i omega x} dx.
            # Discrete approx: F_k ~ dx * sum g_j e^{-i 2pi j k / N} = dx * FFT_k.
            # The frequency spacing for FFT is dnu = 1/(N_fft * dx).
            # If we define Fourier transform with angular frequency omega = 2 pi nu,
            # then domega = 2 pi dnu = 2 pi / (N_fft * dx).
            # Integral |F(omega)|^2 domega approx sum |F_k|^2 * domega.
            # So norm2_hat_sq = sum |G|^2 * (2 * pi / nfft).
            
            G2 = np.abs(G)**2
            norm2_hat_sq = np.sum(G2) * (2 * np.pi / nfft)
            
            G4 = G2**2
            norm4_hat = np.sum(G4) * (2 * np.pi / nfft)
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Sign determination
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
    # Expanded bounds to ensure coverage
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-1, 1), (-1, 1), (-1, 1)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    # Use multiple seeds to avoid local minima
    for seed in [1, 7, 42, 100, 999]:
        try:
            result = differential_evolution(
                objective_from_params, bounds, 
                seed=seed, maxiter=200, popsize=25,
                tol=1e-9, polish=False,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.9
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
        # Try Nelder-Mead with tighter tolerances
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        # Run polish from the best DE result
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
            best_ratio = local_result.fun
            
        # Also try a second polish from a slightly perturbed point to escape potential flat spots
        # Perturb the best solution found
        perturb = np.random.normal(0, 0.1, 3)
        start2 = np.array([c0, c4, c8]) + perturb
        local_result2 = minimize(
            local_obj, 
            start2,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if local_result2.fun < best_ratio:
            c0, c4, c8 = local_result2.x
            best_ratio = local_result2.fun

        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
