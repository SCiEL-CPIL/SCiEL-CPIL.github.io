import numpy as np
from scipy.optimize import minimize

def construct():
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = np.ones_like(x)
    H1 = 2 * x
    H2 = 4 * x**2 - 2
    H3 = 8 * x**3 - 12 * x
    H4 = 16 * x**4 - 48 * x**2 + 12
    H5 = 32 * x**5 - 160 * x**3 + 120 * x
    H6 = 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    H7 = 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H9 = 512 * x**9 - 10752 * x**7 + 60480 * x**5 - 120960 * x**3 + 60480 * x
    H10 = 1024 * x**10 - 30720 * x**8 + 268800 * x**6 - 705600 * x**4 + 604800 * x**2 - 151200
    H11 = 2048 * x**11 - 77824 * x**9 + 896000 * x**7 - 3326400 * x**5 + 4804800 * x**3 - 2217600 * x
    H12 = 4096 * x**12 - 204800 * x**10 + 3225600 * x**8 - 18662400 * x**6 + 40840320 * x**4 - 39916800 * x**2 + 10395000
    
    gauss = np.exp(-x**2 / 2)
    
    def objective(params):
        a0, a4, a8 = params
        if np.any(np.abs(params) > 1e6):
            return 1e10
            
        a12 = (a0 - 3 * a4 + 105 * a8) / 10395.0
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        g = P * gauss
        g2 = g * g
        norm2 = np.sum(g2) * dx
        if norm2 < 1e-15:
            return 1e10
        
        x2g2 = np.sum(x * x * g2) * dx
        xg2 = np.sum(x * g2) * dx
        var_x = x2g2 / norm2 - (xg2 / norm2) ** 2
        if var_x < 0:
            var_x = 0.0

        # Analytical derivative
        P_prime = 4 * a4 * H3 + 16 * a8 * H7 + 24 * a12 * H11
        gp = (P_prime - x * P) * gauss
        gp2 = gp * gp
        gp2_sum = np.sum(gp2) * dx
        
        prod = var_x * (gp2_sum / norm2)
        if prod < 0:
            return 1e10
        return np.sqrt(prod)
    
    x0_candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    ]
    
    best_val = np.inf
    best_x = x0_candidates[0]
    
    for x0 in x0_candidates:
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue
        
    return best_x
