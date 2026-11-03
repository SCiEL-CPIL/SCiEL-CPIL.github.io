import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct, idct

N = 600
TIME_LIMIT = 145  # Safety margin for 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def param_to_step(params):
    """
    Convert parameters to a non-negative step function.
    Parameters are DCT-II coefficients.
    We use a subset of low-frequency modes to ensure smoothness and non-negativity potential.
    To strictly enforce non-negativity during optimization, we use a softplus transformation
    on the reconstructed signal before normalization.
    """
    M = len(params)
    # Reconstruct using DCT-II
    coeffs_full = np.zeros(N)
    coeffs_full[:M] = params
    
    f_raw = idct(coeffs_full, type=2, norm='ortho')
    
    # Softplus to ensure non-negativity and differentiability
    # f = alpha * log(1 + exp(f_raw / alpha))
    alpha = 0.1
    # Numerical stability for softplus
    f_pos = alpha * np.log1p(np.exp(np.clip(f_raw / alpha, -50, 50)))
    
    # Normalize to sum 1
    s = np.sum(f_pos)
    if s < 1e-15:
        # Fallback to uniform
        f_pos = np.ones(N) / N
    else:
        f_pos = f_pos / s
        
    return f_pos

def objective(params):
    """
    Objective function for optimization.
    Returns the normalized max autoconvolution.
    """
    f = param_to_step(params)
    return compute_score(f)

def run_de_optimization(M, time_budget):
    """
    Run Differential Evolution for global optimization.
    """
    start_time = time.time()
    
    # Define bounds for DCT coefficients
    # Low frequency modes can be larger, high frequency smaller
    bounds = []
    for i in range(M):
        # Scale bounds inversely with frequency to encourage smooth solutions
        bound_val = 10.0 / (1 + i * 0.1)
        bounds.append((-bound_val, bound_val))
        
    # Initial population size
    popsize = 15
    
    # DE options
    # maxiter is tricky with time budget, so we rely on callback or just let it run and check time
    # However, DE in scipy doesn't have a direct time limit. We will run a fixed number of iterations
    # and hope it fits, or use a wrapper.
    # Given N=600 and M~20, one eval is fast (~ms).
    # DE with popsize 15, maxiter 50 -> 750 evals. Should be fast.
    
    try:
        res = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=popsize,
            tol=1e-6,
            polish=False,  # We will do our own polish
            init='sobol',  # Better spread
            # callback to check time? DE doesn't support time-based stop easily.
            # We will just run it. If it takes too long, the outer loop will handle it.
            # To be safe, let's limit iterations.
        )
        
        best_params = res.x
        best_score = res.fun
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to a random point or zero
        best_params = np.zeros(M)
        best_score = objective(best_params)
        
    return best_params, best_score

def run_local_refinement(params0, time_budget):
    """
    Run Nelder-Mead simplex method for local refinement.
    """
    start_time = time.time()
    
    try:
        res = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if res.fun < np.inf:
            return res.x, res.fun
        else:
            return params0, objective(params0)
            
    except Exception as e:
        print(f"Local refinement failed: {e}")
        return params0, objective(params0)

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Global search with Differential Evolution on DCT coefficients (M=20)
    # 2. Local refinement with Nelder-Mead on the same coefficients
    
    # Choose M based on time budget
    # M=20 is a good balance of flexibility and optimization dimension
    M = 20
    
    # Phase 1: Global Search (DE)
    # Allocate 60% of time to DE
    de_time_budget = 0.6 * TIME_LIMIT
    
    # Run DE
    params_de, score_de = run_de_optimization(M, de_time_budget)
    
    elapsed = time.time() - start_time
    best_params = params_de
    best_score = score_de
    
    # Phase 2: Local Refinement (Nelder-Mead)
    # Allocate remaining time minus safety margin
    remaining_time = TIME_LIMIT - elapsed - 5.0
    
    if remaining_time > 10.0:
        params_nm, score_nm = run_local_refinement(params_de, remaining_time)
        
        if score_nm < best_score:
            best_params = params_nm
            best_score = score_nm
            
    # Final Check: Try a few other random starts if time permits?
    # DE usually covers this, but a second NM from a different start might help if DE got stuck.
    # Let's stick to the main path to ensure we finish.
    
    # Convert best params to step function
    f_best = param_to_step(best_params)
    
    # Final cleanup
    f_best = np.maximum(f_best, 0.0)
    s = np.sum(f_best)
    if s > 1e-15:
        f_best = f_best / s
        
    # Ensure it's a list of floats
    return [float(x) for x in f_best]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
