import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """
    Design a broadband antireflection coating using a 5-layer stack and Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = 55.0

    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # 5-layer stack: L, H, L, H, L
    materials = ['L', 'H', 'L', 'H', 'L']
    n_vec = np.array([n_L, n_H, n_L, n_H, n_L])
    n_layers = len(materials)

    lambda_c = 0.5 * (band[0] + band[1])
    t_guess = lambda_c / (4.0 * n_vec)
    
    # Ensure initial guess is valid
    if np.sum(t_guess) > max_total_nm:
        t_guess = t_guess * (max_total_nm / np.sum(t_guess))
    t_guess = np.clip(t_guess, 0, max_layer_nm)

    wavelengths = np.linspace(band[0], band[1], 401)

    def compute_reflectance(thicknesses):
        n = n_vec
        t = thicknesses
        delta = 2.0 * np.pi * np.outer(n, t) / wavelengths[np.newaxis, :]
        
        M = np.eye(2, dtype=complex)
        for j in range(n_layers):
            c = np.cos(delta[j, :])
            s = np.sin(delta[j, :])
            
            M00 = M[0,0]*c + M[0,1]*(1j*n[j]*s)
            M01 = M[0,0]*(1j/n[j]*s) + M[0,1]*c
            M10 = M[1,0]*c + M[1,1]*(1j*n[j]*s)
            M11 = M[1,0]*(1j/n[j]*s) + M[1,1]*c
            
            M[0,0] = M00
            M[0,1] = M01
            M[1,0] = M10
            M[1,1] = M11
            
        M11 = M[0,0]
        M12 = M[0,1]
        M21 = M[1,0]
        M22 = M[1,1]
        
        Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
        R = np.abs((n_0 - Y) / (n_0 + Y))**2
        
        return np.mean(R) * 100.0

    # Transform variables: t = max_layer_nm * sigmoid(x)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def logit(t):
        t = np.clip(t, 1e-6, max_layer_nm - 1e-6)
        return np.log(t / (max_layer_nm - t))
        
    x_guess = logit(t_guess)

    def objective(x):
        t = max_layer_nm * sigmoid(x)
        if np.sum(t) > max_total_nm:
            return 100.0 # Penalty
        return compute_reflectance(t)

    try:
        res = minimize(objective, x_guess, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success or res.fun < objective(x_guess):
            t_opt = max_layer_nm * sigmoid(res.x)
            if np.sum(t_opt) <= max_total_nm and np.all(t_opt >= 0) and np.all(t_opt <= max_layer_nm):
                return materials, list(t_opt)
    except Exception:
        pass
        
    return materials, list(t_guess)
