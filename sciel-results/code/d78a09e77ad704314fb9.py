"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    # Hermite values at 0
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def compute_norms(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        coeffs = [c0, c4, c8, c12]
        degrees = [0, 4, 8, 12]
        
        # Define g(x) and hat_g(xi)
        # g(x) = sum c_k H_k(x) exp(-x^2/2)
        # hat_g(xi) = sqrt(2*pi) * sum c_k (-i)^k H_k(xi) exp(-xi^2/2)
        
        def g_x(x):
            val = 0.0
            for c, d in zip(coeffs, degrees):
                val += c * eval_hermite(d, x) * np.exp(-0.5 * x**2)
            return val
        
        def hat_g_xi(xi):
            val = 0.0
            for c, d in zip(coeffs, degrees):
                # (-i)^d
                phase = (-1j)**d
                val += c * phase * eval_hermite(d, xi) * np.exp(-0.5 * xi**2)
            return val
        
        # Integrate |g|^2, |g|^4, |hat_g|^2, |hat_g|^4
        # Use quad for precision. Limits -15 to 15 should be sufficient for exp(-x^2/2)
        
        norm2_sq, _ = quad(lambda x: g_x(x)**2, -15, 15, limit=100)
        norm4_val, _ = quad(lambda x: abs(g_x(x))**4, -15, 15, limit=100)
        
        hat_norm2_sq, _ = quad(lambda xi: abs(hat_g_xi(xi))**2, -15, 15, limit=100)
        hat_norm4_val, _ = quad(lambda xi: abs(hat_g_xi(xi))**4, -15, 15, limit=100)
        
        if norm2_sq < 1e-12 or hat_norm2_sq < 1e-12:
            return 1e10
            
        norm2 = np.sqrt(norm2_sq)
        hat_norm2 = np.sqrt(hat_norm2_sq)
        norm4 = norm4_val ** 0.25
        hat_norm4 = hat_norm4_val ** 0.25
        
        # The ratio in the seed code was:
        # (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
        ratio = (norm4 * hat_norm4) / (norm2 * hat_norm2)
        
        return ratio

    def objective_from_log_params(params, signs):
        m0, m4, m8 = params
        c0 = signs[0] * np.exp(m0)
        c4 = signs[1] * np.exp(m4)
        c8 = signs[2] * np.exp(m8)
        return compute_norms(c0, c4, c8)

    best_ratio = 1e10
    best_params = None
    
    # Iterate over all 8 sign combinations
    for s0 in [-1, 1]:
        for s4 in [-1, 1]:
            for s8 in [-1, 1]:
                signs = [s0, s4, s8]
                
                # Initial guess: log(1) = 0
                x0 = [0.0, 0.0, 0.0]
                
                try:
                    res = minimize(
                        objective_from_log_params, 
                        x0, 
                        args=(signs,),
                        method='L-BFGS-B',
                        bounds=[(-5, 5), (-5, 5), (-5, 5)],
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    
                    if res.fun < best_ratio:
                        best_ratio = res.fun
                        best_params = (res.x, signs)
                except Exception:
                    pass
    
    if best_params is not None:
        m0, m4, m8 = best_params[0]
        signs = best_params[1]
        c0 = signs[0] * np.exp(m0)
        c4 = signs[1] * np.exp(m4)
        c8 = signs[2] * np.exp(m8)
        return np.array([c0, c4, c8])
    else:
        # Fallback
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
