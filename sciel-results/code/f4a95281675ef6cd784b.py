import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.fft import fft, fftfreq

def construct():
    def score_fn(coeffs):
        a0, a4, a8 = coeffs
        a12 = -(a0 + 12.0 * a4 + 168.0 * a8) / 4032.0
        
        N = 65536
        x_max = 8.0
        x = np.linspace(-x_max, x_max, N)
        dx = x[1] - x[0]
        
        H0 = np.ones_like(x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        g = P * np.exp(-x**2 / 2.0)
        
        A_g = np.sum(g**2) * dx
        
        g_hat = fft(g) * dx
        A_g_hat = np.sum(np.abs(g_hat)**2) * dx
        
        return A_g * A_g_hat
    
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    result = minimize(score_fn, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    
    return result.x
