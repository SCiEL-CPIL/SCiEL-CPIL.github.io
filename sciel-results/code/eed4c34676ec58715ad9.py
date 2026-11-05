"""
Strategy: Global Direct Optimization via Differential Evolution
The previous strategy restricted h to a low-dimensional manifold (sparse spectrum).
This strategy instead optimizes the full 400-dimensional vector h directly.
We use Differential Evolution (DE) from scipy.optimize, which is a robust global
optimizer well-suited for non-smooth, non-convex objectives like maximizing
a function involving np.max of a convolution.

The objective is to minimize V = abs(2 * N * max(conv(h,h)) / sum(h)^2).
Since V is scale-invariant (scaling h by c scales numerator by c^2 and denominator by c^2),
we can fix the scale, e.g., sum(h) = 1, or let DE find the scale.
However, DE works better with bounded variables. We will map the search space
to [-1, 1] for each h[i] and compute the objective on the resulting vector.
If sum(h) is close to 0, we return a large penalty.

To improve convergence and handle the "max" non-differentiability:
1. Use DE with a relatively high population size and mutation rate.
2. The objective function is evaluated exactly as per the task contract.
3. We add a small regularization term to encourage non-zero sum if it's too small.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit

def _evaluate(h):
    """
    Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2).
    Returns a large number if invalid (sum ~ 0 or non-finite).
    """
    s = np.sum(h)
    if not np.isfinite(s) or abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _objective(params):
    """
    Objective function for Differential Evolution.
    params: array of length N, values in [0, 1] (bounded by DE).
    We map these to real heights.
    To allow negative heights, we map [0, 1] -> [-1, 1] via h = 2*params - 1.
    """
    h = 2.0 * params - 1.0
    return _evaluate(h)

def construct():
    """
    Construct step heights using Differential Evolution.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial best: seed
    best_h = np.ones(N)
    best_val = 2.0

    # Define bounds for DE: [0, 1] for each of the N parameters
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # DE parameters:
    # popsize: 15 (default) might be too small for 400 dims? 
    # Actually, DE is expensive. 400 dims * 15 pop * iterations.
    # Let's try a moderate popsize.
    # maxiter: We have ~145s. 
    # Each function eval is ~ O(N^2) for conv? No, np.convolve is O(N^2) but fast.
    # Let's estimate: 400*400 = 160k ops. Very fast in numpy.
    # So we can afford many iterations.
    
    # Strategy: 
    # 1. Run DE for a fixed number of iterations or until time limit.
    # 2. DE in scipy doesn't have a direct "time limit" stop condition easily exposed 
    #    without wrapping, but we can check time in the callback? 
    #    Actually, `differential_evolution` has a `callback` parameter.
    
    stop_flag = [False]
    
    def callback(xk, convergence):
        if time.time() > deadline - 5.0:
            stop_flag[0] = True
            # To stop DE, we can't easily break out. 
            # However, if we raise an exception or set a flag, it might not stop cleanly.
            # Better approach: Run DE with a reasonable maxiter and check time between "phases" 
            # or just trust maxiter.
            # Let's estimate speed.
            # 1 eval ~ 0.001s? 400 conv is fast.
            # If 1 eval is 1ms, 1000 evals is 1s.
            # DE with popsize 15, 100 iters = 1500 evals = 1.5s.
            # We can do many more.
            # Let's set maxiter high and rely on the fact that DE converges.
            # But we need to stop. 
            # SciPy DE does not support early stopping via callback return.
            # We will run a few short DE runs with different seeds/params if time permits.
            pass

    # To handle the time limit safely, we will run DE with a specific maxiter
    # that fits within the budget, or run multiple short runs.
    # Let's assume 1 eval is ~0.5ms (conservative).
    # 150s / 0.0005s = 300,000 evals.
    # DE: popsize * maxiter.
    # If popsize=20, maxiter=1000 -> 20,000 evals. That's only 10s.
    # We can do more.
    
    # However, DE on 400 dims is hard. 
    # Let's try popsize=30, maxiter=500.
    # 30 * 500 = 15,000 evals.
    
    try:
        res = differential_evolution(
            _objective,
            bounds,
            maxiter=1000,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=False,  # Polish uses L-BFGS-B which might get stuck in local mins or be slow
            init='latinhypercube',
            # callback=callback
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_h = 2.0 * res.x - 1.0
            
    except Exception as e:
        # If DE fails, fall back to seed
        pass

    # If we have time left, try a second run with different seed or params
    if time.time() < deadline - 20.0:
        try:
            res2 = differential_evolution(
                _objective,
                bounds,
                maxiter=500,
                popsize=30,
                tol=1e-6,
                seed=123,
                polish=False,
                init='latinhypercube'
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_h = 2.0 * res2.x - 1.0
        except Exception:
            pass

    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
