"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, minimize

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using a real-valued signal constructed from the real part of a chirp 
    (product of Zadoff-Chu and another chirp) modulated by a raised-cosine window,
    with an optimized DC offset.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    t = np.arange(N, dtype=np.float64)
    n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)

    def get_h_ac(params):
        """
        params: [gamma, phi, beta, win_param]
        gamma: Chirp rate (alpha - mu)
        phi: Linear phase slope (-mu)
        beta: Global phase
        win_param: Window shape parameter (0=rect, 1=Hann)
        """
        gamma, phi, beta, win_param = params
        
        # Phase: pi * gamma * t^2 / N + pi * phi * t / N + beta
        phase = np.pi * gamma * t * t / N + np.pi * phi * t / N + beta
        signal = np.cos(phase)
        
        # Raised cosine window
        window = 1.0 - 0.5 * win_param + 0.5 * win_param * np.cos(np.pi * n_norm)
        
        h_ac = signal * window
        return h_ac

    def optimize_dc(x):
        """Finds the optimal DC offset c to add to x to minimize the value."""
        S_x = np.sum(x)
        C_x = np.convolve(x, x)
        U = np.convolve(x, np.ones(N))
        T = np.convolve(np.ones(N), np.ones(N))
        
        def obj(c):
            conv_c = C_x + 2*c*U + c**2*T
            m = np.max(conv_c)
            s = S_x + N*c
            if abs(s) < 1e-12:
                return float('inf')
            val = abs(2 * N * m / (s * s))
            return val
        
        best_c = 0.0
        best_obj = obj(0.0)
        
        # Test some large c values to find the basin
        for c_test in [1, 5, 10, 20, 50]:
            v = obj(c_test)
            if v < best_obj:
                best_obj = v
                best_c = c_test
                
        # Refine around the best found
        if best_c > 0:
            res = minimize_scalar(obj, bounds=(max(0, best_c*0.5), best_c*1.5), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        else:
             # If best was 0 or negative, try a wider range
             res = minimize_scalar(obj, bounds=(-10, 10), method='bounded', options={'xatol': 1e-8})
             if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x

        return x + best_c, best_obj

    def objective(params):
        try:
            h_ac = get_h_ac(params)
            h, val = optimize_dc(h_ac)
            return val
        except Exception:
            return float('inf')

    # Differential Evolution for global search
    # Bounds:
    # gamma: -10 to 10
    # phi: -5 to 5
    # beta: 0 to 2*pi
    # win_param: 0 to 1
    
    bounds = [(-10, 10), (-5, 5), (0, 2*np.pi), (0, 1)]
    
    # Run DE
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=50, 
        tol=1e-6, 
        popsize=10
    )
    
    best_params = res.x
    best_val_found = res.fun
    
    # Refine with Powell
    def refine_objective(params):
        return objective(params)
        
    res2 = minimize(
        refine_objective,
        best_params,
        method='Powell',
        options={'maxiter': 100, 'maxfev': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    if res2.fun < best_val_found:
        best_val_found = res2.fun
        best_params = res2.x
        
    if best_val_found < best_val:
        h_ac = get_h_ac(best_params)
        best_h, _ = optimize_dc(h_ac)
        return best_h
        
    return best_h
# EVOLVE-BLOCK-END
