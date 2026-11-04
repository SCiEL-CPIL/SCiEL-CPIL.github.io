"""Optimized program for the third autocorrelation inequality task using chirp representation."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def chirp_to_heights(params):
    """Convert chirp parameters (phase, chirp_rate) to step heights.
    
    h[j] = Re(exp(i * (phase + chirp_rate * j^2 / N)))
    This creates a signal with linearly varying frequency (chirp) that
    decorrelates the autocorrelation sidelobes.
    """
    phase = params[0]
    chirp_rate = params[1]
    j = np.arange(N)
    # Quadratic phase for chirp
    phase_arr = phase + chirp_rate * (j ** 2) / N
    h = np.cos(phase_arr)
    return h


def objective(params):
    """Objective function for optimization."""
    h = chirp_to_heights(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin

    # Phase 1: Differential evolution for global search over chirp parameters
    # Parameters: [initial_phase, chirp_rate]
    bounds = [(-np.pi, np.pi), (-10.0, 10.0)]
    
    def de_objective(params):
        h = chirp_to_heights(params)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s ** 2))
    
    best_params = np.array([0.0, 0.0])
    best_val = 2.0  # Seed value
    
    # Try a few different random seeds for diversity
    for seed in [42, 123, 777]:
        if time.time() > DEADLINE - 60.0:
            break
        
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                seed=seed,
                maxiter=200,
                tol=1e-4,
                popsize=15,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False,
                init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
        
        if time.time() > DEADLINE - 30.0:
            break
    
    # Phase 2: Nelder-Mead local refinement from best DE solution
    if time.time() < DEADLINE - 15.0:
        try:
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if nm_result.fun < best_val:
                best_val = nm_result.fun
                best_params = nm_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try additional local refinements from nearby points
    if time.time() < DEADLINE - 10.0:
        for offset in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            if time.time() > DEADLINE - 5.0:
                break
            try:
                perturbed = best_params.copy()
                perturbed[1] += offset
                nm_result = minimize(
                    objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                if nm_result.fun < best_val:
                    best_val = nm_result.fun
                    best_params = nm_result.x.copy()
            except Exception:
                continue
    
    # Generate final heights from best parameters
    h = chirp_to_heights(best_params)
    
    # Ensure sum is not zero and finite
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h)):
        return np.ones(N)
    
    # Final validation
    final_val = compute_value(h)
    if final_val >= 2.0:
        # If chirp didn't help, fall back to seed
        return np.ones(N)
    
    return h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
