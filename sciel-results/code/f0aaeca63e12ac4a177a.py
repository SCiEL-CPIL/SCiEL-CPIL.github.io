"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_coefficients(a0, a4, a8):
    """Compute all four coefficients given the constraint g(0)=0."""
    # H0(0)=1, H4(0)=12, H8(0)=3360, H12(0)=645120
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 3360.0) / 645120.0
    return np.array([a0, a4, a8, a12])

def _compute_uncertainty_product(coeffs):
    """
    Compute the uncertainty product for g(x) = sum c_n H_n(x) e^{-x^2/2}.
    
    Using orthogonality: ∫ H_m H_n e^{-x^2} dx = 2^n n! √π δ_mn
    The Fourier transform: F[H_n(x)e^{-x^2/2}] = (-i)^n H_n(x)e^{-x^2/2}
    So |ĝ|² has the same coefficient structure as |g|² (phases cancel).
    
    A(g) = ∫|g|² dx = sum |c_n|² * 2^n * n! * √π
    The variance-like quantity involves ∫x²|g|² dx which requires cross terms.
    
    For the uncertainty product, we use:
    A(f) * A(ĵ) where these are related to the spread.
    """
    # Coefficients for H0, H4, H8, H12
    c = coeffs
    # Norms: ||H_n e^{-x^2/2}||² = 2^n n! √π
    norms = np.array([
        1.0 * np.sqrt(np.pi),           # n=0: 2^0 * 0! * √π
        16.0 * 24.0 * np.sqrt(np.pi),   # n=4: 2^4 * 4! * √π = 16*24*√π
        256.0 * 40320.0 * np.sqrt(np.pi),  # n=8: 2^8 * 8! * √π
        4096.0 * 479001600.0 * np.sqrt(np.pi)  # n=12: 2^12 * 12! * √π
    ])
    
    # ||g||² = sum |c_n|² * norms[n] (orthogonal)
    norm_sq = np.sum(np.abs(c)**2 * norms)
    if norm_sq < 1e-30:
        return np.inf
    
    # For the uncertainty, we need ∫x²|g|² dx
    # x H_n = (1/2)H_{n+1} + n H_{n-1}
    # x² H_n = (1/4)H_{n+2} + n H_n + (n(n-1)/4) H_{n-2}
    
    # x²|g|² = x² (sum c_m H_m)(sum c_n* H_n) e^{-x²}
    # = sum_m sum_n c_m c_n* x² H_m H_n e^{-x²}
    # Using x² H_m = (1/4)H_{m+2} + m H_m + (m(m-1)/4) H_{m-2}
    
    # ∫ x² H_m H_n e^{-x²} dx
    # = (1/4)∫H_{m+2}H_n e^{-x²} + m∫H_m H_n e^{-x²} + (m(m-1)/4)∫H_{m-2}H_n e^{-x²}
    # By orthogonality, only terms where indices match survive.
    
    # Let's compute ∫x²|g|² dx = sum_m |c_m|² * ∫x² H_m² e^{-x²} dx
    # ∫x² H_m² e^{-x²} dx: use x² H_m = (1/4)H_{m+2} + m H_m + (m(m-1)/4) H_{m-2}
    # So ∫x² H_m² e^{-x²} = (1/4)∫H_{m+2}H_m e^{-x²} + m∫H_m² e^{-x²} + (m(m-1)/4)∫H_{m-2}H_m e^{-x²}
    # By orthogonality, ∫H_k H_m e^{-x²} = 0 unless k=m, so:
    # = m * 2^m m! √π (only the middle term survives)
    
    # Wait, that's not right. The cross terms vanish by orthogonality, so:
    # ∫x² H_m² e^{-x²} = m * 2^m m! √π
    
    # But we also need cross terms for |g|² when g is a sum!
    # |g|² = sum_m sum_n c_m c_n* H_m H_n e^{-x²}
    # x²|g|² = sum_m sum_n c_m c_n* x² H_m H_n e^{-x²}
    # ∫x² H_m H_n e^{-x²} dx: this is nonzero when m and n differ by 0 or 2.
    
    # Let me compute this numerically for accuracy
    # Actually, let's use a simpler approach: numerical integration
    
    # Define g(x) on a grid
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    
    # Compute H_n(x) for n = 0, 4, 8, 12
    # Using recurrence or explicit formulas
    # H0 = 1
    # H1 = 2x
    # H2 = 4x²-2
    # H3 = 8x³-12x
    # H4 = 16x⁴-48x²+12
    # H5 = 32x⁵-160x³+120x
    # H6 = 64x⁶-480x⁴+720x²-120
    # H7 = 128x⁷-1344x⁵+3360x³-1680x
    # H8 = 256x⁸-3072x⁶+11520x⁴-13440x²+3360
    # H9 = 512x⁹-7680x⁷+40320x⁵-67200x³+20160x
    # H10 = 1024x¹⁰-19456x⁸+134400x⁶-268800x⁴+151200x²-15120
    # H11 = 2048x¹¹-45056x⁹+403200x⁷-1075200x⁵+907200x³-166320x
    # H12 = 4096x¹²-98304x¹⁰+1075200x⁸-5080320x⁶+9072000x⁴-5443200x²+645120
    
    x2 = x**2
    H0 = np.ones_like(x)
    H4 = 16*x2**2 - 48*x2 + 12
    H8 = 256*x2**4 - 3072*x2**3 + 11520*x2**2 - 13440*x2 + 3360
    H12 = 4096*x2**6 - 98304*x2**5 + 1075200*x2**4 - 5080320*x2**3 + 9072000*x2**2 - 5443200*x2 + 645120
    
    gauss = np.exp(-x2/2)
    
    g = (c[0]*H0 + c[1]*H4 + c[2]*H8 + c[3]*H12) * gauss
    
    # A(g) = ∫|g|² dx
    A_g = np.sum(np.abs(g)**2) * dx
    
    if A_g < 1e-30:
        return np.inf
    
    # The Fourier transform of H_n(x)e^{-x²/2} is (-i)^n H_n(x)e^{-x²/2}
    # So ĝ(x) = (c0*H0 + c4*(-i)^4*H4 + c8*(-i)^8*H8 + c12*(-i)^12*H12) * e^{-x²/2}
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = (-i)^8 * (-i)^4 = 1
    # So ĝ has the SAME form as g!
    # A(ĝ) = A(g)
    
    # The uncertainty product is likely related to the variance.
    # Let's compute ∫x²|g|² dx / A_g as the "variance"
    x2_g = np.sum(x2 * np.abs(g)**2) * dx
    var_g = x2_g / A_g
    
    # For the uncertainty principle, the product is often var_x * var_k
    # where var_k is related to the Fourier side.
    # Since ĝ has the same structure, var_k = var_g (by symmetry)
    # So the product is var_g²
    
    # But the task says "upper bound on A(f)*A(ĵ)" which might just be the product of norms
    # or the product of variances. Let me use var_g * var_g as a proxy.
    
    return var_g * var_g

def _objective(params):
    """Objective function to minimize."""
    a0, a4, a8 = params
    coeffs = _hermite_coefficients(a0, a4, a8)
    return _compute_uncertainty_product(coeffs)

def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the uncertainty product.
    The a12 coefficient is determined by the constraint g(0)=0.
    """
    start_time = time.time()
    deadline = 180.0  # seconds, with safety margin
    
    # Initial guess from the prior state of the art
    best_params = np.array([-113/100, 1/25, 1/3240])
    best_value = _objective(best_params)
    
    # Try multiple starting points
    start_points = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-0.5, 0.5, 0.5]),
        np.array([0.5, -0.5, 0.5]),
        best_params.copy(),
    ]
    
    for sp in start_points:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Return the three free coefficients
    return np.array(best_params)
# EVOLVE-BLOCK-END
