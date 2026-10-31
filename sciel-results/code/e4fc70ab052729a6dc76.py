"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
import time

def construct():
    # Start timing to ensure we finish within 180s
    start_time = time.time()
    
    # Use a well-chosen grid for accurate integration
    # The Gaussian decays quickly, so we don't need to go too far out
    x = np.linspace(-12, 12, 6000)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)  # = 1
    H4_0 = eval_hermite(4, 0)  # = -12
    H8_0 = eval_hermite(8, 0)  # = 1680
    H12_0 = eval_hermite(12, 0)  # = -199584
    
    # Precompute Gaussian
    gauss = np.exp(-0.5 * x**2)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
            
            # Check if g is essentially zero
            if np.max(np.abs(g)) < 1e-10:
                return 1e10
            
            # Compute norms using trapezoidal rule
            g2 = g**2
            norm2_sq = np.trapz(g2, x)
            if norm2_sq < 1e-15:
                return 1e10
                
            g4 = g2**2
            norm4 = np.trapz(g4, x)
            if norm4 < 1e-30:
                return 1e10
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT with proper scaling
            # The Fourier transform of g(x) = sum c_n H_n(x) e^{-x^2/2}
            # We use the definition: F(ξ) = ∫ g(x) e^{-iξx} dx
            # Using FFT: if g_k = g(x_k), then F(ξ_j) ≈ dx * sum_k g_k e^{-iξ_j x_k}
            # For FFT, we need to be careful with the frequency grid
            
            n = len(g)
            # Pad to next power of 2 for efficiency, but not too large
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            # Cap nfft to avoid memory issues
            nfft = min(nfft, 2**18)
            
            # Zero-pad g for FFT
            g_pad = np.zeros(nfft)
            g_pad[:n] = g
            
            # Compute FFT
            G = np.fft.fft(g_pad) * dx
            
            # The frequency grid: freq_k = k / (nfft * dx) for k = 0, ..., nfft/2, then negative
            # Actually, np.fft.fftfreq gives frequencies in cycles per unit
            # We need angular frequency for the integral
            # The integral ∫|F(ξ)|^2 dξ where F is the Fourier transform
            # Using Parseval: ∫|g(x)|^2 dx = (1/2π) ∫|F(ξ)|^2 dξ
            # So ∫|F(ξ)|^2 dξ = 2π ∫|g(x)|^2 dx
            
            # But we need ∫|F(ξ)|^4 dξ
            # Let's compute it numerically
            freq = np.fft.fftfreq(nfft, d=dx)  # cycles per unit
            # Convert to angular frequency
            omega = 2 * np.pi * freq
            
            # The spacing in omega
            domega = omega[1] - omega[0] if nfft > 1 else 1.0
            
            # Compute |G|^2 and |G|^4
            G2 = np.abs(G)**2
            G4 = G2**2
            
            # Integrate over all frequencies
            norm2_hat_sq = np.trapz(G2, omega)
            norm4_hat = np.trapz(G4, omega)
            
            if norm2_hat_sq < 1e-15 or norm4_hat < 1e-30:
                return 1e10
            
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use smooth sign transition to help the optimizer
            # sign = tanh(s) * 0.9 + 0.1 * sign(s) to avoid exact zeros
            sign0 = np.tanh(s0) * 0.8 + 0.2 * np.sign(s0)
            sign4 = np.tanh(s4) * 0.8 + 0.2 * np.sign(s4)
            sign8 = np.tanh(s8) * 0.8 + 0.2 * np.sign(s8)
            
            # Handle case where sign is zero
            if abs(sign0) < 0.01:
                sign0 = 0.01 if s0 >= 0 else -0.01
            if abs(sign4) < 0.01:
                sign4 = 0.01 if s4 >= 0 else -0.01
            if abs(sign8) < 0.01:
                sign8 = 0.01 if s8 >= 0 else -0.01
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -5 to 4 (coefficients from ~0.0067 to ~54.6)
    # s_i: sign parameter, range from -2 to 2
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-2, 2), (-2, 2), (-2, 2)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    # Use multiple seeds with different strategies
    seeds = [42, 123, 456, 789, 1011]
    
    for seed in seeds:
        # Check time
        if time.time() - start_time > 120:
            break
            
        try:
            result = differential_evolution(
                objective_from_params, bounds, 
                seed=seed, maxiter=200, popsize=25,
                tol=1e-10, polish=False,
                init='sobol',
                mutation=(0.4, 0.9),
                recombination=0.7,
                constraints=[]
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_params = result.x
        except Exception:
            continue
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = np.tanh(s0) * 0.8 + 0.2 * np.sign(s0)
        sign4 = np.tanh(s4) * 0.8 + 0.2 * np.sign(s4)
        sign8 = np.tanh(s8) * 0.8 + 0.2 * np.sign(s8)
        
        if abs(sign0) < 0.01:
            sign0 = 0.01 if s0 >= 0 else -0.01
        if abs(sign4) < 0.01:
            sign4 = 0.01 if s4 >= 0 else -0.01
        if abs(sign8) < 0.01:
            sign8 = 0.01 if s8 >= 0 else -0.01
            
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        # Try multiple local optimization starting points
        best_local_ratio = best_ratio
        best_local_coeffs = [c0, c4, c8]
        
        # Starting points: the current best and some perturbations
        start_points = [
            [c0, c4, c8],
            [c0 * 1.1, c4 * 1.1, c8 * 1.1],
            [c0 * 0.9, c4 * 0.9, c8 * 0.9],
            [c0, c4 * 1.1, c8],
            [c0, c4, c8 * 1.1],
            [c0 * 1.1, c4, c8],
        ]
        
        for sp in start_points:
            if time.time() - start_time > 160:
                break
                
            try:
                local_result = minimize(
                    local_obj, 
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                
                if local_result.fun < best_local_ratio:
                    best_local_ratio = local_result.fun
                    best_local_coeffs = local_result.x
            except Exception:
                continue
        
        if best_local_ratio < best_ratio:
            c0, c4, c8 = best_local_coeffs
        
        return np.array([c0, c4, c8])
    else:
        # Fallback: try a few hand-tuned values
        # Based on the structure, we want coefficients that make g(0)=0 and minimize the ratio
        # Let's try some reasonable values
        candidates = [
            [-1.13, 0.04, 1.0 / 3240.0],
            [-1.0, 0.1, 0.001],
            [-0.5, 0.05, 0.0005],
            [-2.0, 0.08, 0.002],
        ]
        
        best_candidate = candidates[0]
        best_candidate_ratio = 1e10
        
        for cand in candidates:
            ratio = compute_ratio(cand[0], cand[1], cand[2])
            if ratio < best_candidate_ratio:
                best_candidate_ratio = ratio
                best_candidate = cand
        
        return np.array(best_candidate)

# EVOLVE-BLOCK-END
