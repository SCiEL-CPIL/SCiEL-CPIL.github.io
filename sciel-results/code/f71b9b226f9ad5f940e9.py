import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    k = 10
    
    # Gauss-Legendre quadrature in cos(theta): 70 points
    x, w = np.polynomial.legendre.leggauss(70)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    dphi = 2*np.pi / n_phi
    
    # Compute spherical harmonics Y_m for m = -10..10
    m_vals = np.arange(-k, k+1)
    n_points = 70 * 130
    
    Y = np.zeros((n_points, 21), dtype=np.float64)
    
    for idx, m in enumerate(m_vals):
        am = abs(m)
        N_m = np.sqrt(21.0/(4.0*np.pi) * float(factorial(10-am))/float(factorial(10+am)))
        P = lpmv(am, 10, x)
        
        if m == 0:
            Y_col = np.tile(N_m * P, (n_phi, 1)).T
        elif m > 0:
            cos_m_phi = np.cos(m * phi)
            Y_col = (np.sqrt(2) * N_m * P[:, np.newaxis] * cos_m_phi[np.newaxis, :]).flatten()
        else:
            sin_am_phi = np.sin(am * phi)
            Y_col = (np.sqrt(2) * N_m * P[:, np.newaxis] * sin_am_phi[np.newaxis, :]).flatten()
        
        Y[:, idx] = Y_col
    
    # Quadrature weights
    w_full = np.tile(w * dphi, (n_phi, 1)).flatten()
    
    def objective(c):
        f = Y @ c
        f4 = np.sum(w_full * np.abs(f)**4)
        return f4**0.25
    
    def objective_grad(c):
        f = Y @ c
        # d/dc (sum w_i |f_i|^4)^(1/4) = (1/4) (sum w_i |f_i|^4)^(-3/4) * 4 * sum w_i |f_i|^3 f_i * Y_i
        # = (sum w_i |f_i|^3 f_i * Y_i) / (sum w_i |f_i|^4)^(3/4)
        f4 = np.sum(w_full * np.abs(f)**4)
        if f4 < 1e-30:
            return np.zeros(21)
        grad = Y.T @ (w_full * np.abs(f)**3 * f)
        grad /= f4**0.75
        return grad
    
    # Constraint: ||c||_2 = 1
    constraint = {'type': 'eq', 'fun': lambda c: np.sum(c**2) - 1.0}
    
    c0 = np.ones(21)
    c0 /= np.linalg.norm(c0)
    
    best_c = c0.copy()
    best_val = objective(c0)
    
    np.random.seed(42)
    for trial in range(100):
        if trial == 0:
            c_start = c0.copy()
        else:
            c_start = np.random.randn(21)
            c_start /= np.linalg.norm(c_start)
        
        result = minimize(objective, c_start, jac=objective_grad,
                         method='SLSQP', constraints=constraint,
                         options={'maxiter': 1000, 'ftol': 1e-14})
        
        if result.fun < best_val:
            c_opt = result.x / np.linalg.norm(result.x)
            val = objective(c_opt)
            if val < best_val:
                best_val = val
                best_c = c_opt
    
    return best_c
