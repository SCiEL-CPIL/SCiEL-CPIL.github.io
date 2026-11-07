"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Golden Section Search-based Coordinate Descent optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Track function calls
        self._call_count = 0

        def f(x):
            self._call_count += 1
            return float(func(np.asarray(x, dtype=float)))

        # Initial point: try a few random starts and pick the best
        # Use a slightly larger initial set to ensure we are in the basin
        n_init = min(15, budget // 10)
        x_best = None
        fx_best = np.inf
        for _ in range(n_init):
            x_cand = rng.uniform(-4.5, 4.5, d)
            fcand = f(x_cand)
            if fcand < fx_best:
                fx_best = fcand
                x_best = x_cand.copy()

        if self._call_count >= budget:
            return x_best, fx_best

        # Golden Section Search for 1D minimization
        def golden_1d(x0, axis, lo, hi, max_evals):
            """
            Minimize f along a single axis using Golden Section Search.
            x0: current point
            axis: which axis to minimize along
            lo, hi: bounds for the coordinate
            max_evals: maximum number of function evaluations
            Returns (x_new, f_new)
            """
            if self._call_count + max_evals > budget:
                max_evals = max(1, budget - self._call_count)
            
            x = x0.copy()
            
            # Golden ratio
            invphi = (np.sqrt(5.0) - 1.0) / 2.0
            invphi2 = invphi * invphi
            
            a = lo
            b = hi
            
            # Initial points
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            
            x[axis] = c
            fc = f(x)
            x[axis] = d
            fd = f(x)
            
            # We used 2 evaluations.
            # Loop until we run out of budget or max_evals
            # We track evaluations used in this specific 1D search
            evals_used = 2
            
            while evals_used < max_evals and self._call_count < budget:
                if fc < fd:
                    # Minimum is in [a, d]
                    b = d
                    d = c
                    fd = fc
                    c = b - invphi * (b - a)
                    x[axis] = c
                    fc = f(x)
                    evals_used += 1
                else:
                    # Minimum is in [c, b]
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    x[axis] = d
                    fd = f(x)
                    evals_used += 1
                    
            # Determine the best point
            if fc < fd:
                x[axis] = c
                fx_final = fc
            else:
                x[axis] = d
                fx_final = fd
                
            return x, fx_final

        # Main optimization loop
        # We perform coordinate descent passes.
        # Each pass iterates through all dimensions.
        # We use a dynamic number of evaluations per axis based on remaining budget.
        
        # To be efficient, we'll do passes until budget is nearly exhausted.
        # Each axis search costs ~10-20 evaluations.
        # Let's target 10 evaluations per axis initially, reducing as we go.
        
        # We'll use a while loop that breaks when we can't afford a full pass
        # or when we are very close to the end.
        
        while self._call_count < budget - 10: # Leave 10 for final tweaks
            # Determine how many evaluations we can afford per axis in this pass
            # We want to do at least 5 evals per axis to make progress
            remaining = budget - self._call_count
            # If we have enough for a full pass with 10 evals per axis, do it.
            # Otherwise, do a partial pass or reduce evals.
            
            # Simple strategy: Try to do a full pass with ~10 evals per axis.
            # If budget is tight, do fewer evals.
            evals_per_axis = 10
            if remaining < d * 5:
                break # Not enough for a meaningful pass
            
            # If we have very little left, reduce evals per axis
            if remaining < d * 15:
                evals_per_axis = max(5, remaining // d - 1)
            
            for axis in range(d):
                if self._call_count >= budget - 5:
                    break
                
                # Define bounds
                # Start with full bounds, but we can shrink them if we are confident.
                # For robustness, keep [-5, 5] but Golden Section converges quickly.
                lo = -5.0
                hi = 5.0
                
                x_new, fx_new = golden_1d(x_best, axis, lo, hi, evals_per_axis)
                
                if fx_new < fx_best:
                    x_best = x_new
                    fx_best = fx_new

        # Final refinement: use any remaining budget for a few more single-axis steps
        remaining = budget - self._call_count
        if remaining > 0:
            # Do as many single-axis refinements as possible
            # Use small evals per axis (e.g., 3-5)
            evals_small = max(3, min(5, remaining // d))
            while self._call_count + evals_small < budget:
                for axis in range(d):
                    if self._call_count + evals_small > budget:
                        break
                    x_new, fx_new = golden_1d(x_best, axis, -5.0, 5.0, evals_small)
                    if fx_new < fx_best:
                        x_best = x_new
                        fx_best = fx_new

        return x_best, fx_best
# EVOLVE-BLOCK-END
