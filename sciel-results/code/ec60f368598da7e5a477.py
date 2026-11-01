import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 10s safety margin
    
    # Hermite polynomials H0, H4, H8, H12
    # Using scipy.special.eval_hermite
    def H(n, x):
        return eval_hermite(n, x)
    
    # Constraint: g(0) = 0
    # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = -12
    # H8(0) = 1344
    # H12(0) = -199584
    h0_0 = 1.0
    h4_0 = -12.0
    h8_0 = 1344.0
    h12_0 = -199584.0
    
    def get_c12(c0, c4, c8):
        # c0*h0_0 + c4*h4_0 + c8*h8_0 + c12*h12_0 = 0
        return -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0

    # Setup Gauss-Hermite quadrature for integrals of form int e^{-x^2} f(x) dx
    # We need high accuracy. 100 points should be sufficient for degree 12 polynomial squared (degree 24)
    # times x^2 (degree 2) -> degree 26. GH quadrature with N points is exact for degree 2N-1.
    # So N=14 is exact for degree 27. Let's use N=50 to be safe and fast.
    x_nodes, w_weights = np.polynomial.hermite.hermgauss(50)
    
    def objective(params):
        c0, c4, c8 = params
        c12 = get_c12(c0, c4, c8)
        
        # Evaluate P(x) at nodes
        P_x = c0*H(0, x_nodes) + c4*H(4, x_nodes) + c8*H(8, x_nodes) + c12*H(12, x_nodes)
        
        # |g(x)|^2 = e^{-x^2} P(x)^2
        # Integral is sum(w_i * P(x_i)^2) because the e^{-x^2} is in the weight
        # Wait, hermgauss weights are for int e^{-x^2} f(x) dx.
        # So int |g|^2 dx = int e^{-x^2} P(x)^2 dx = sum(w_i * P(x_i)^2)
        g_sq = P_x**2
        norm_x = np.sum(w_weights * g_sq)
        
        if norm_x < 1e-10:
            return 1e10 # Penalty for zero norm
            
        x2_g_sq = x_nodes**2 * g_sq
        var_x_num = np.sum(w_weights * x2_g_sq)
        var_x = var_x_num / norm_x
        
        # Frequency domain
        # g_hat(xi) is related to P(i xi).
        # |g_hat(xi)|^2 = e^{-xi^2} |P(i xi)|^2
        # We need int e^{-xi^2} |P(i xi)|^2 dxi
        # P(i xi) = c0*H0(i xi) + c4*H4(i xi) + c8*H8(i xi) + c12*H12(i xi)
        # H_n(i xi) is complex.
        
        xi_nodes = x_nodes # Same grid
        P_i_xi = c0*H(0, 1j*xi_nodes) + c4*H(4, 1j*xi_nodes) + c8*H(8, 1j*xi_nodes) + c12*H(12, 1j*xi_nodes)
        
        g_hat_sq = np.abs(P_i_xi)**2
        norm_xi = np.sum(w_weights * g_hat_sq)
        
        if norm_xi < 1e-10:
            return 1e10
            
        xi2_g_hat_sq = xi_nodes**2 * g_hat_sq
        var_xi_num = np.sum(w_weights * xi2_g_hat_sq)
        var_xi = var_xi_num / norm_xi
        
        # Product of standard deviations
        prod = np.sqrt(var_x * var_xi)
        
        # Check deadline
        if time.time() > deadline:
            # We can't easily abort mid-eval in a simple callback, 
            # but the optimizer will check after this step.
            pass
            
        return prod

    # Initial guess from seed
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    # Try a few starting points to ensure we find the global minimum
    best_val = 1e10
    best_params = [c0_init, c4_init, c8_init]
    
    starting_points = [
        [c0_init, c4_init, c8_init],
        [0, 0, 0], # Will result in c12=0, g(x)=0? No, if c0=c4=c8=0, c12=0, g=0. Norm 0.
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1, 1, 1],
        [-1, 1, -1]
    ]
    
    for sp in starting_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # If we still have time, refine with Powell
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_params, method='Powell', options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Return the three coefficients c0, c4, c8
    return np.array(best_params)
