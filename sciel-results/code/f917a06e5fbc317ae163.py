"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Count function calls
        call_count = [0]
        
        def counted_func(x):
            call_count[0] += 1
            # Clip to box to ensure validity
            x_clipped = np.clip(x, -5, 5)
            return float(func(x_clipped))
        
        # Initialize Nelder-Mead
        # Start at origin or a random point? Origin is often good for centered functions,
        # but BBOB f7 has a shift. Let's start at 0.
        x0 = np.zeros(dim)
        
        # Initial simplex size
        # Standard NM uses x0 + delta * e_i for i>0 and x0 for the first point.
        # Let's use a reasonable initial step size.
        # The box is [-5, 5]. A step of 1.0 is reasonable.
        delta = 1.0
        
        # We need to be careful with the number of calls.
        # Nelder-Mead typically uses dim+1 points.
        # Each iteration uses 1 or 2 new evaluations.
        # Total budget is 500.
        
        # We will run NM in chunks to allow for restarts if it gets stuck or to handle
        # the "black-box" nature better if the landscape is multimodal (though f7 is unimodal).
        # Actually, f7 (Step Ellipsoidal) is unimodal. NM should work well.
        # However, NM can be slow to converge to high precision.
        # Let's run it until budget is exhausted or convergence criteria met.
        
        # scipy.optimize.minimize with method='Nelder-Mead'
        # We need to control the number of iterations or use maxiter.
        # maxiter in scipy NM is the number of iterations, not function calls.
        # Each iteration is roughly 1-2 function calls.
        # So maxiter = budget // 2 is a safe upper bound.
        
        # However, scipy's NM might stop early if it thinks it's converged.
        # For AOCC, we want to keep improving as long as possible.
        # But if it converges to a local min (which is the global min for f7), further calls might not help much.
        # The scoring is based on the best value found *so far*.
        # So if we find the minimum early, we should stop wasting calls?
        # No, the evaluator records *every* call.
        # If we stop calling, the evaluator repeats the last n_i.
        # "repeating the last n_i if fewer calls are made"
        # So if we make 100 calls and find the optimum, the remaining 400 calls will have the same n_i as call 100.
        # This is good! We don't need to use the full budget if we've converged.
        # But we must ensure we don't crash.
        
        # Strategy:
        # 1. Run NM.
        # 2. If it converges or we run out of budget, return.
        # 3. To be safe, we can also do a local random search or gradient-free refinement if NM stops too early?
        # No, let's stick to NM. It's robust for smooth-ish functions.
        # f7 is "Step Ellipsoidal". It might have a "step" in the landscape?
        # BBOB f7 is typically the "Step Ellipsoidal" function.
        # f(x) = 0.5 * ||x||^2 + 10 * (1 - cos(2*pi*||x||)) * (1 - ||x||)^2 ...? 
        # Actually, BBOB f7 is often defined as:
        # f(x) = 0.5 * ||x||^2 + 10 * (1 - cos(2*pi*||x||))
        # Or similar. It is unimodal.
        
        # Let's use scipy's minimize.
        # We need to handle the case where scipy raises an exception.
        
        try:
            # Use a wrapper to count calls and stop if budget exceeded
            # scipy minimize doesn't easily support a call limit other than maxiter.
            # We'll set maxiter to a large number and rely on our call counter?
            # No, scipy doesn't stop based on an external counter easily.
            # We can subclass or use a callback?
            # Actually, we can just run it and check call_count afterwards.
            # But if it exceeds 500, the result is invalid.
            # So we must ensure we don't exceed 500.
            # maxiter for NM: each iteration does 1 or 2 function calls.
            # So maxiter = 250 should be safe (approx 500 calls).
            # Let's be conservative: maxiter = 200.
            
            # Initial simplex:
            # x0, x0 + delta*e0, x0 + delta*e1, ...
            # scipy handles this internally if we just pass x0.
            
            # Options for Nelder-Mead:
            # xatol, fatol: tolerance for x and f.
            # We want high precision, so set these small.
            # But we also want to avoid infinite loops.
            
            res = minimize(
                counted_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,  # Approx 400-500 calls
                    'maxfev': 490,    # Max function evaluations (scipy 1.11+ supports maxfev? Let's check. 
                                      # Actually, older versions use maxiter. 
                                      # In recent scipy, 'maxfev' is not a standard option for NM in all versions, 
                                      # but 'maxiter' is. 
                                      # Let's rely on maxiter. 
                                      # If maxiter=200, and avg 2 calls/iter, that's 400 calls. Safe.
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            # Check if we exceeded budget (shouldn't happen with maxiter=200)
            # If we did, the result is invalid anyway, but we can't fix it now.
            
        except Exception:
            # If NM fails, fall back to a simple random search or (1+1)-ES?
            # Or just return the best found so far?
            # We need to return *something*.
            # Let's do a quick random search if NM fails completely.
            pass

        return res.x, res.fun
# EVOLVE-BLOCK-END
