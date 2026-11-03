import numpy as np
import time
from scipy.optimize import differential_evolution

N = 50

def _score_raw(h):
    """
    Calculate the raw score R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is scale invariant, so we can assume ||h||_1 = 1.
    Then R(h) = ||h||_2^2 / ||h*h||_inf.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to L1 = 1
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    
    # Autoconvolution
    # The peak of the continuous autoconvolution is approx by DELTA * max(conv(h, h)).
    # However, the constant DELTA cancels out in the ratio if we consider the discrete approximation 
    # consistent with the evaluator's likely definition of "supremum norm of the autoconvolution" 
    # for step functions. 
    # Let's verify the definition:
    # L2^2 = DELTA * sum(h_i^2)
    # L1 = DELTA * sum(h_i)
    # Supremum of autoconvolution = DELTA * max(conv(h, h))
    # Ratio = (DELTA * L2_sq_discrete) / (DELTA * L1_discrete * DELTA * MaxConv_discrete)
    #       = (1/DELTA) * (L2_sq_discrete / (L1_discrete * MaxConv_discrete))
    # Since 1/DELTA is constant, we maximize the discrete ratio.
    
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
        
    return l2_sq / max_conv

def _objective(x):
    """
    Objective function for minimization.
    x is a vector of length N.
    We maximize _score_raw, so we minimize -_score_raw.
    """
    # Ensure non-negativity and sum to 1
    x = np.maximum(x, 0)
    s = np.sum(x)
    if s < 1e-15:
        return 0.0
    x_norm = x / s
    return -_score_raw(x_norm)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (5s buffer before 150s kill)
    
    # Initial best guess
    # A sparse vector often performs better.
    # Let's start with a random sparse vector to give DE a good initial population bias if needed,
    # or just use uniform. DE is global, so it should find good structures.
    
    # Strategy: Run Differential Evolution on the full 50-dim space.
    # Bounds: [0, 1] for each height. The objective normalizes internally.
    # To help DE, we can initialize with some sparsity.
    
    # Differential Evolution settings
    # popsize: 15 * N is standard, but N=50 means 750 individuals.
    # Each evaluation involves a convolution (O(N^2) or O(N log N) but small N).
    # 750 evals per generation.
    # Max 150s.
    # Let's estimate speed.
    
    # We'll run DE with a time limit.
    # We can use a callback to check time.
    
    def time_callback(strategy, convergence, population, **kwargs):
        if time.time() > deadline - 2.0:
            raise TimeoutError("Time limit reached")
            
    # Initial population strategy:
    # DE usually uses uniform random in bounds.
    # We can provide a custom initial population with some sparsity to guide it.
    # But DE is robust. Let's try standard first.
    
    # To improve chances, let's run a few DE instances with different seeds/parameters
    # or just one long run. Given the budget, one well-tuned run is safer.
    
    # Let's try a "sparse-aware" initialization.
    # We can generate random vectors with many zeros.
    # scipy.optimize.differential_evolution accepts an initial population via 'init'
    # but it must be a matrix of shape (popsize, N).
    
    popsize = 30  # Smaller popsize to allow more generations within time
    # Actually, default is 15*N = 750. That might be too slow if evaluations are expensive.
    # Let's check evaluation cost.
    # Convolution of length 50 is very fast.
    # 750 evals * ~1000 generations = 750,000 evals.
    # 150s / 750,000 = 0.2ms per eval.
    # np.convolve(50, 50) is extremely fast, likely < 10us.
    # So 750,000 evals should take < 10s.
    # We can afford a larger popsize or more iterations.
    
    # Let's use popsize=50 (2.5 * N) and maxiter=1000.
    # Total evals ~ 50,000. Very fast.
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    try:
        # Differential Evolution
        # bounds: [(0, 1)] * N
        # We want to maximize, so we minimize negative score.
        
        res = differential_evolution(
            _objective,
            bounds=[(0, 1)] * N,
            maxiter=2000,
            popsize=50,
            tol=1e-8,
            seed=42,
            callback=time_callback,
            polish=True  # Use L-BFGS-B at the end for local refinement
        )
        
        if res.success:
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            s = np.sum(h_opt)
            if s > 0:
                h_opt /= s
            score_opt = _score_raw(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt
        else:
            # Even if not "successful" in convergence, the best point found might be good
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            s = np.sum(h_opt)
            if s > 0:
                h_opt /= s
            score_opt = _score_raw(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt
                
    except TimeoutError:
        pass
    except Exception:
        pass

    # If time remains, try a second DE run with different parameters or seed
    if time.time() < deadline - 10:
        try:
            # Try a more aggressive sparsity bias by using a different initialization or penalty?
            # DE doesn't easily support sparsity penalties without modifying objective.
            # We can modify the objective to slightly penalize large support?
            # Or just run another DE with different seed.
            
            res2 = differential_evolution(
                _objective,
                bounds=[(0, 1)] * N,
                maxiter=2000,
                popsize=50,
                tol=1e-8,
                seed=123,
                callback=time_callback,
                polish=True
            )
            
            if res2.success:
                h_opt2 = res2.x
                h_opt2 = np.maximum(h_opt2, 0)
                s2 = np.sum(h_opt2)
                if s2 > 0:
                    h_opt2 /= s2
                score_opt2 = _score_raw(h_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_h = h_opt2
            else:
                h_opt2 = res2.x
                h_opt2 = np.maximum(h_opt2, 0)
                s2 = np.sum(h_opt2)
                if s2 > 0:
                    h_opt2 /= s2
                score_opt2 = _score_raw(h_opt2)
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_h = h_opt2
                    
        except TimeoutError:
            pass
        except Exception:
            pass
            
    # Final check
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_score_raw(h)}")
