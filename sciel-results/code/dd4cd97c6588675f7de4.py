import numpy as np
from scipy.optimize import minimize

def construct():
    x = np.linspace(-8, 8, 10001)
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
        
        gp = np.gradient(g, dx)
        gp2 = gp * gp
        gp2_sum = np.sum(gp2) * dx
        gpp = np.gradient(gp, dx)
        gpp2 = gpp * gpp
        gpp2_sum = np.sum(gpp2) * dx
        
        # Use a combination as proxy for the bound
        return np.sqrt(max(var_x, 0) * max(gp2_sum / norm2, 0))
    
    x0 = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    if result.fun < objective(x0):
        return np.array(result.x)
    return x0
