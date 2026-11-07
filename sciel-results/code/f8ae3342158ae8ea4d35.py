"""Multi-objective optimizer on RE34: Tchebycheff scalarization with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Tchebycheff scalarization optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for Tchebycheff scalarization
        # We use the known reference point from the task description
        ref = np.array([1705.0, 12.0, 0.3])
        
        # However, for Tchebycheff, we usually want a point that is worse than all expected points.
        # The task says "minimizing all 3 objectives".
        # Let's use a dynamic reference or a fixed one.
        # Since we don't know the exact bounds of the Pareto front, we can estimate.
        # Or just use the reference point provided in scoring? No, that's for hypervolume.
        # For Tchebycheff, we need a "utopia" or "ideal" point.
        # Let's use a large value or estimate from initial samples.
        
        used = 0
        X = []
        F = []

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        # Phase 1: Initial Sampling to estimate ideal point and get diverse starts
        n_init = 30
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = lo[d] + (i + rng.random()) / n_init * (hi[d] - lo[d])
            x, f = evaluate(x)
            X.append(x)
            F.append(f)
            
        if used >= budget:
            return X

        F_arr = np.array(F)
        # Estimate ideal point (minimums)
        ideal = F_arr.min(axis=0)
        # Add a small margin to ensure it's strictly better or at least representative
        # Actually, Tchebycheff uses: min max_i (w_i * (f_i - z_i^*))
        # where z^* is the ideal point.
        
        # Phase 2: Tchebycheff Scalarization
        # Generate weight vectors. For 3 objectives, we can use a grid or random.
        # To cover the front well, we use a set of weights.
        n_weights = 30
        # Generate weights on the simplex
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(n_obj))
            # Ensure no weight is too small to focus on extremes
            weights.append(w)
            
        # Also add some extreme weights to find knee points
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
            
        # Shuffle weights
        rng.shuffle(weights)
        
        # For each weight, run a local search (Nelder-Mead)
        # We have budget - used evaluations left.
        # Each Nelder-Mead run takes some evaluations.
        # Let's limit each run to a max number of function calls.
        # scipy.optimize.minimize with method='Nelder-Mead'
        
        # We need to be careful with the budget.
        # Let's reserve some budget for exploration if time permits, but Tchebycheff is fast.
        
        for w in weights:
            if used >= budget - 5: # Leave some margin
                break
            if time.time() > deadline:
                break
                
            # Choose a starting point. 
            # A good strategy is to pick the point that minimizes the scalarized objective
            # given the current archive, or just a random point from the archive.
            # Let's pick the best point from the current archive for this weight.
            if F:
                F_arr = np.array(F)
                # Compute Tchebycheff objective for all current points
                # obj = max_i (w_i * (f_i - ideal_i))
                # Since we want to minimize, and f_i are positive, and ideal_i is min,
                # (f_i - ideal_i) >= 0.
                diffs = F_arr - ideal
                # Handle case where ideal might be slightly off due to sampling
                diffs = np.maximum(diffs, 0) 
                tcheb_vals = np.max(w * diffs, axis=1)
                start_idx = np.argmin(tcheb_vals)
                x0 = X[start_idx].copy()
            else:
                x0 = rng.uniform(lo, hi)
                
            # Define scalarized objective
            def tcheb_obj(x):
                f = np.asarray(problem(np.clip(x, lo, hi)), dtype=np.float64)
                # Update global used counter
                nonlocal used
                used += 1
                diffs = f - ideal
                diffs = np.maximum(diffs, 0)
                return np.max(w * diffs)

            # Run Nelder-Mead
            # We need to limit the number of function calls to avoid exhausting budget.
            # maxfev is the key parameter.
            # Let's allocate roughly (budget - used) / remaining_weights
            remaining_budget = budget - used
            remaining_weights = len(weights) - weights.index(w)
            maxfev = max(10, int(remaining_budget / max(1, remaining_weights)))
            
            try:
                res = minimize(tcheb_obj, x0, method='Nelder-Mead', 
                               options={'maxfev': maxfev, 'xatol': 1e-6, 'fatol': 1e-8})
                # res.fun is the best value found, res.x is the best x
                # But we need to record all evaluations? No, the task says "Every returned objective vector is recorded".
                # The evaluator calls problem(x) and records the result.
                # So we don't need to manually record, just ensure we don't exceed budget.
                # However, we should add the best found point to our archive for future starts.
                if res.success or used < budget:
                    x_best = np.clip(res.x, lo, hi)
                    # Evaluate once more to get the exact f for the archive?
                    # The minimize function already evaluated it.
                    # But we don't have the f vector from res directly, only the scalar value.
                    # So we must evaluate it again to get the vector.
                    # This costs 1 extra evaluation.
                    if used < budget:
                        x_best, f_best = evaluate(x_best)
                        X.append(x_best)
                        F.append(f_best)
                        # Update ideal point
                        ideal = np.minimum(ideal, f_best)
            except Exception:
                pass

        return X
# EVOLVE-BLOCK-END
