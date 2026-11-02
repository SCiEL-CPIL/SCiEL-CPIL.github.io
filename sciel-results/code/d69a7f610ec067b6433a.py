"""
Optimized Hermite coefficients for uncertainty bound.
Uses fast numerical integration on a grid and multi-start Nelder-Mead.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 180.0

    # 1. Generate Hermite Polynomials H_0, H_4, H_8, H_12
    # Recurrence: H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    # We represent polynomials as coefficient arrays [c_0, c_1, ..., c_n] for c_0 + c_1 x + ... + c_n x^n
    
    def hermite_coeffs(n):
        if n == 0:
            return np.array([1.0])
        if n == 1:
            return np.array([0.0, 2.0])
        
        # H_{n-2} and H_{n-1}
        h_nm2 = hermite_coeffs(n-2)
        h_nm1 = hermite_coeffs(n-1)
        
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        # 2x H_{n-1} shifts coefficients by 1
        term1 = np.zeros(len(h_nm1) + 1)
        term1[1:] = 2.0 * h_nm1
        
        term2 = -2.0 * (n-1) * h_nm2
        # Pad term2 to match length
        if len(term2) < len(term1):
            term2 = np.pad(term2, (0, len(term1) - len(term2)), 'constant')
        elif len(term1) < len(term2):
            term1 = np.pad(term1, (0, len(term2) - len(term1)), 'constant')
            
        return term1 + term2

    H0_c = hermite_coeffs(0)
    H4_c = hermite_coeffs(4)
    H8_c = hermite_coeffs(8)
    H12_c = hermite_coeffs(12)

    # 2. Setup Grid
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2.0)
    
    # Evaluate Polynomials on Grid
    # np.polyval expects coefficients in descending order [c_n, ..., c_0]
    # My hermite_coeffs returns [c_0, ..., c_n]
    H0 = np.polyval(H0_c[::-1], x)
    H4 = np.polyval(H4_c[::-1], x)
    H8 = np.polyval(H8_c[::-1], x)
    H12 = np.polyval(H12_c[::-1], x)
    
    # Values at 0 for constraint
    H0_0 = H0_c[0] # 1
    H4_0 = H4_c[0] # 12
    H8_0 = H8_c[0] # 1680
    H12_0 = H12_c[0] # -10395? Let's check sign.
    # H_12(0) = -2(11) H_10(0). H_10(0) = -2(9) H_8(0) = -18(1680) = -30240.
    # H_12(0) = -22(-30240) = 665280.
    # Let's trust the polynomial generation.

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        # a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    def objective(coeffs):
        """
        Compute the upper bound on C4:
        J = (||g||_4 * ||F(g)||_4) / (||g||_2 * ||F(g)||_2)
        """
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        # g(x)
        poly_g = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        g = poly_g * gauss
        
        # F(g)(x)
        # i^0=1, i^4=1, i^8=1, i^12=-1
        poly_fg = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
        fg = poly_fg * gauss
        
        # Norms
        g2_sq = np.sum(g**2) * dx
        fg2_sq = np.sum(fg**2) * dx
        
        if g2_sq < 1e-15 or fg2_sq < 1e-15:
            return 1e10
            
        norm2_g = np.sqrt(g2_sq)
        norm2_fg = np.sqrt(fg2_sq)
        
        g4_4th = np.sum(g**4) * dx
        fg4_4th = np.sum(fg**4) * dx
        
        if g4_4th < 1e-15 or fg4_4th < 1e-15:
            return 1e10
            
        norm4_g = g4_4th ** 0.25
        norm4_fg = fg4_4th ** 0.25
        
        return (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

    # 3. Optimization
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed.copy()
    best_val = objective(seed)
    
    rng = np.random.RandomState(42)
    
    # Multi-start Nelder-Mead
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
            
        if i == 0:
            x0 = seed.copy()
        else:
            # Random start in a reasonable neighborhood
            # Scale decreases as we go deeper? Or just fixed?
            # The space is likely not huge.
            scale = 1.0
            x0 = seed + rng.normal(0, scale, size=3)
            
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Final refinement
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
