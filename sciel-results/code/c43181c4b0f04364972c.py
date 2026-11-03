import numpy as np
from scipy.special import eval_hermite
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize
import time

# Precompute matrices
indices = [0, 4, 8, 12]
n = len(indices)

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0

def compute_matrices():
    G = np.zeros((n, n))
    K = np.zeros((n, n))
    L = np.zeros((n, n))
    
    for i in range(n):
        k_i = indices[i]
        for j in range(n):
            k_j = indices[j]
            
            def integrand_G(x):
                return eval_hermite(k_i, x) * eval_hermite(k_j, x) * np.exp(-2 * np.pi * x**2)
            
            def integrand_K(x):
                return (x**2) * eval_hermite(k_i, x) * eval_hermite(k_j, x) * np.exp(-2 * np.pi * x**2)
            
            def integrand_L(x):
                if k_i == 0 and k_j == 0:
                    return 0.0
                d_hi = 0.0 if k_i == 0 else 2 * k_i * eval_hermite(k_i - 1, x)
                d_hj = 0.0 if k_j == 0 else 2 * k_j * eval_hermite(k_j - 1, x)
                return d_hi * d_hj * np.exp(-2 * np.pi * x**2)
            
            G[i, j], _ = quad(integrand_G, -np.inf, np.inf, limit=100)
            K[i, j], _ = quad(integrand_K, -np.inf, np.inf, limit=100)
            L[i, j], _ = quad(integrand_L, -np.inf, np.inf, limit=100)
            
    return G, K, L

G_mat, K_mat, L_mat = compute_matrices()

def get_c12(c0, c4, c8):
    return -(c0 + 12*c4 + 1680*c8) / H12_0

def objective(params):
    c0, c4, c8 = params
    c12 = get_c12(c0, c4, c8)
    v = np.array([c0, c4, c8, c12])
    
    N0 = v @ G_mat @ v
    Nx = v @ K_mat @ v
    Nd = v @ L_mat @ v
    
    if N0 <= 1e-12:
        return 1e10
        
    return np.sqrt(Nx * Nd) / N0

def construct():
    start = time.time()
    
    res = differential_evolution(
        objective, 
        [(-5, 5), (-5, 5), (-1, 1)], 
        maxiter=50, 
        popsize=15, 
        seed=123,
        tol=1e-9
    )
    
    best_params = res.x
    best_score = res.fun
    
    if time.time() - start < 150:
        res2 = minimize(objective, best_params, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 500})
        if res2.fun < best_score:
            best_params = res2.x
            best_score = res2.fun
            
        if time.time() - start < 180:
             res3 = differential_evolution(
                objective, 
                [(-10, 10), (-10, 10), (-2, 2)], 
                maxiter=30, 
                popsize=10, 
                seed=999,
                tol=1e-9
             )
             if res3.fun < best_score:
                 best_params = res3.x
                 best_score = res3.fun
    
    return np.array(best_params)
