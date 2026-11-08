"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Hermite polynomials (physicist's convention)
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    
    def H12(x):
        return 4096*x**12 - 73728*x**10 + 460800*x**8 - 1290240*x**6 + 1574400*x**4 - 725760*x**2 + 1034880
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 1034880.0
    
    def get_c12(c0, c4, c8):
        """Derive c12 to satisfy g(0) = 0"""
        return -(c0 + H4_0*c4 + H8_0*c8) / H12_0
    
    def g_func(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        return (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * np.exp(-x**2/2.0)
    
    def compute_C4(c0, c4, c8, n_points=20000):
        """
        Compute the uncertainty bound C4 = A(f) * A(Fourier(f))
        where A(f) = ||f||_1 and A(F) = ||Fourier(f)||_1
        """
        x = np.linspace(-12, 12, n_points)
        dx = x[1] - x[0]
        f = g_func(x, c0, c4, c8)
        
        # L1 norm of f
        L1_f = np.sum(np.abs(f)) * dx
        
        # FFT for Fourier transform
        n = len(f)
        nfft = 1 << (n-1).bit_length()
        f_padded = np.zeros(nfft)
        f_padded[:n] = f
        F = np.fft.fft(f_padded) * dx
        
        # L1 norm of Fourier transform
        L1_F = np.sum(np.abs(F)) * dx
        
        return L1_f * L1_F
    
    # Start with the seed coefficients
    seed_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_coeffs = seed_coeffs.copy()
    best_C4 = compute_C4(*seed_coeffs)
    
    # Try differential evolution for global optimization
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            def objective_de(params):
                return compute_C4(params[0], params[1], params[2])
            
            result_de = differential_evolution(
                objective_de, bounds,
                maxiter=50, popsize=15,
                seed=42, tol=1e-8,
                polish=True
            )
            
            if result_de.fun < best_C4:
                best_C4 = result_de.fun
                best_coeffs = result_de.x
        except Exception:
            pass
    
    # Refine with Nelder-Mead
    if time.time() - start_time < deadline - 30:
        try:
            def objective_nm(params):
                return compute_C4(params[0], params[1], params[2])
            
            result_nm = minimize(
                objective_nm, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result_nm.fun < best_C4:
                best_C4 = result_nm.fun
                best_coeffs = result_nm.x
        except Exception:
            pass
    
    # Additional local refinement with tighter tolerances
    if time.time() - start_time < deadline - 15:
        try:
            def objective_final(params):
                return compute_C4(params[0], params[1], params[2], n_points=30000)
            
            result_final = minimize(
                objective_final, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            
            if result_final.fun < best_C4:
                best_C4 = result_final.fun
                best_coeffs = result_final.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
