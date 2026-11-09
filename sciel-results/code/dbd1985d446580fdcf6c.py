"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import cobyla
import time


class Optimizer:
    """Cobyla-based optimizer for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Cobyla is deterministic given a start point.
        # We use a small number of random restarts to ensure we don't get stuck
        # in local minima if the landscape is tricky, though f12 is unimodal.
        # However, the rotation and shift mean the "center" is unknown.
        # A single run from a random point might be sufficient for f12, 
        # but to be safe and utilize the budget if Cobyla converges quickly,
        # we can try a few different starting points or just one robust one.
        
        # Given the 500 call budget and Cobyla's efficiency, one run might finish in <100 calls.
        # We can use the remaining budget for a second restart if the first one finishes early,
        # or just let it run. But cobyla doesn't have a strict "call budget" in the same way.
        # We need to wrap func to count calls and stop if we exceed budget.
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            if calls_made > self.budget:
                # Return a very large number to discourage further moves
                return 1e10
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial point: center of the box is a safe bet for unimodal functions,
        # but the minimum is shifted. Random is better to explore.
        # Let's try the origin first, as it's often close in these benchmarks if shift is small,
        # or a random point. The prompt says minimizer is in [-4,4]^5.
        # A random start is more robust against shifts.
        
        x0 = rng.uniform(-5, 5, self.dim)
        
        # Cobyla parameters:
        # rhobeg: initial step size. 1.0 is reasonable for [-5,5].
        # tol: termination tolerance. 1e-12 is good for precision.
        
        try:
            res = cobyla(
                wrapped_func, 
                x0, 
                bounds=[(-5.0, 5.0)] * self.dim,
                rhobeg=1.0,
                tol=1e-12
            )
        except Exception:
            # If cobyla fails, return best found so far
            if best_x is None:
                best_x = x0
                best_f = func(x0)
            return best_x, best_f

        # If cobyla finishes before using the full budget, we could potentially
        # restart from the result with a smaller rhobeg or different start,
        # but for f12 (unimodal), one good run is usually sufficient.
        # However, to maximize AOCC, we want progress early.
        # Cobyla provides progress as it iterates.
        
        # The return value of __call__ is ignored, but we must return something.
        # The evaluator records every call to func.
        
        if best_x is None:
             # Fallback if no calls were made (unlikely)
             best_x = x0
             best_f = func(x0)
             
        return best_x, best_f
# EVOLVE-BLOCK-END
