import numpy as np
from scipy.optimize import minimize

def construct():
    x = np.linspace(-8, 8, 10001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials (Physicists')
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H12 = 4096 * x**12 - 20480 * x**10 + 322560 * x**8 - 1866240 * x**6 + 4084032 * x**4 - 3991680 * x**2 + 665280
    
    gauss = np.exp(-x**2 / 2)
    
    def objective(params):
        a0, a4, a8 = params
        # Correct constraint for Physicists' Hermite: P(0) = a0*1 + a4*12 + a8*1680 + a12*665280 = 0
        a12 = -(a0 + 12 * a4 + 1680 * a8) / 665280.0
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
        
        # Uncertainty product
        return np.sqrt(max(var_x, 0) * max(gp2_sum / norm2, 0))
    
    # Initial guess: Try to approximate a function with a node at 0.
    # H4 has a large negative curvature at 0? H4(0)=12. H0(0)=1.
    # We need a0 + 12a4 + ... = 0.
    # Let a4 = 1, a0 = -12. a8=0, a12=0.
    # P = -12 H0 + H4 = -12 + 16x^4 - 48x^2 + 12 = 16x^4 - 48x^2 = 16x^2(x^2-3).
    # This has zeros at 0, +/-sqrt(3).
    x0 = np.array([-12.0, 1.0, 0.0])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    if result.fun < objective(x0):
        return np.array(result.x)
    return x0
