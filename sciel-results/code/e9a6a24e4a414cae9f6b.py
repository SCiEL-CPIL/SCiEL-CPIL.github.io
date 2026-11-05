"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def soft_max(x, tau):
    """Compute soft maximum of x."""
    # Clip to avoid overflow in exp
    x_max = np.max(x)
    if x_max > 100:
        x = x - x_max
        val = x_max + tau * np.log(np.sum(np.exp(x / tau)))
    else:
        val = tau * np.log(np.sum(np.exp(x / tau)))
    return val

def objective(params, N, tau, penalty_weight=1e6):
    """
    Objective function for optimization.
    params: [A_1...A_M, f_1...f_M, phi_1...phi_M]
    """
    M = len(params) // 3
    A = params[:M]
    f = params[M:2*M]
    phi = params[2*M:3*M]
    
    # Reconstruct h
    k = np.arange(N)
    h = np.zeros(N)
    for j in range(M):
        h += A[j] * np.cos(2 * np.pi * f[j] * k + phi[j])
    
    S = np.sum(h)
    if abs(S) < 1e-6:
        # Penalize zero sum
        return 1e10 + penalty_weight * (1 - abs(S))
        
    # Compute autocorrelation
    # np.convolve(h, h, 'full') has length 2N-1
    # We only care about the max value
    c = np.convolve(h, h, 'full')
    
    max_c = soft_max(c, tau)
    
    val = 2.0 * N * max_c / (S * S)
    
    return val

def polish(h_init, N, time_limit):
    """
    Polish the h vector directly using L-BFGS-B.
    """
    start_time = time.time()
    
    def obj_direct(h):
        S = np.sum(h)
        if abs(S) < 1e-6:
            return 1e10
        c = np.convolve(h, h, 'full')
        max_c = soft_max(c, 0.01) # Small tau for high precision
        return 2.0 * N * max_c / (S * S)

    # Normalize h to avoid scaling issues during optimization
    h_norm = h_init / np.linalg.norm(h_init) if np.linalg.norm(h_init) > 1e-9 else h_init
    
    # Use L-BFGS-B
    try:
        res = minimize(obj_direct, h_norm, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
        if res.success:
            return res.x
    except Exception:
        pass
        
    return h_norm

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_value = 2.0
    best_h = np.ones(N)
    
    # Number of sinusoids to use
    M = 8
    
    # Number of restarts
    max_restarts = 20
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialize sinusoidal parameters
        # Frequencies in [0, 0.5] to avoid redundancy
        f_init = np.random.uniform(0.01, 0.5, M)
        A_init = np.random.uniform(-1, 1, M)
        phi_init = np.random.uniform(0, 2*np.pi, M)
        
        params_init = np.concatenate([A_init, f_init, phi_init])
        
        # 2. Optimize sinusoidal parameters
        # Use Nelder-Mead for robustness against non-convexity
        tau = 1.0
        try:
            res = minimize(objective, params_init, args=(N, tau), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                best_params = res.x
                val_sin = res.fun
            else:
                # If it fails, try with a smaller tau or different init? 
                # For now, just use the current best params or skip
                best_params = params_init
                val_sin = objective(params_init, N, tau)
                
            # Reconstruct h from optimized params
            M_cur = len(best_params) // 3
            A = best_params[:M_cur]
            f = best_params[M_cur:2*M_cur]
            phi = best_params[2*M_cur:3*M_cur]
            
            k = np.arange(N)
            h_candidate = np.zeros(N)
            for j in range(M_cur):
                h_candidate += A[j] * np.cos(2 * np.pi * f[j] * k + phi[j])
                
            # Check validity
            S = np.sum(h_candidate)
            if abs(S) < 1e-6:
                continue
                
            # 3. Polish
            h_polished = polish(h_candidate, N, time.time())
            
            # Evaluate polished result
            S_p = np.sum(h_polished)
            if abs(S_p) < 1e-6:
                continue
                
            c_p = np.convolve(h_polished, h_polished, 'full')
            max_c_p = np.max(c_p)
            val_p = 2.0 * N * max_c_p / (S_p * S_p)
            
            if val_p < best_value:
                best_value = val_p
                best_h = h_polished
                
        except Exception as e:
            continue
            
    return best_h
# EVOLVE-BLOCK-END
