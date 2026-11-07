# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using Multi-start Local Optimization
    with Tchebycheff scalarization for RE34.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s.
        # We aim to finish well before that.
        deadline = start_time + 300 

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume calculation (not used in optimization directly, 
        # but good to know for context if we were doing EHVI)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Track evaluations
        eval_count = 0
        all_X = []
        all_F = []
        
        # Lock for thread safety (not strictly needed here but good practice)
        # We use a simple counter variable.

        def objective_wrapper(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            eval_count += 1
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        # Generate weight vectors for Tchebycheff scalarization
        # We want to cover the trade-off space.
        # For 3 objectives, we can use a set of predefined weights.
        # A grid of weights on the simplex x+y+z=1, x,y,z >= 0.
        
        # Let's use 10 diverse weight vectors.
        weights = []
        # Simple manual selection for 3 objectives
        # 1. Equal weights
        weights.append(np.array([1.0/3, 1.0/3, 1.0/3]))
        # 2. Emphasizing each objective
        weights.append(np.array([0.7, 0.15, 0.15]))
        weights.append(np.array([0.15, 0.7, 0.15]))
        weights.append(np.array([0.15, 0.15, 0.7]))
        # 3. Intermediate
        weights.append(np.array([0.5, 0.25, 0.25]))
        weights.append(np.array([0.25, 0.5, 0.25]))
        weights.append(np.array([0.25, 0.25, 0.5]))
        # 4. More extreme
        weights.append(np.array([0.8, 0.1, 0.1]))
        weights.append(np.array([0.1, 0.8, 0.1]))
        weights.append(np.array([0.1, 0.1, 0.8]))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        n_restarts = len(weights)
        
        # Initial points: Use random points or specific heuristic points
        # For RE34, the Pareto front is often found near the center or specific corners.
        # Let's use a Latin Hypercube-like sampling for initial points, or just random.
        # Given dim=5, random is fine.
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within run? 
        # The evaluator calls np.random.seed(seed), so we should use np.random directly 
        # or ensure we don't interfere with the global state if possible. 
        # The task says "evaluator calls np.random.seed(seed)". 
        # Using np.random directly is safer to align with the evaluator's seeding.
        
        # We will use np.random for generating initial points.
        
        # Distribute budget
        # Leave some budget for potential refinement or extra random samples
        # 200 budget, 10 restarts. 
        # If we give 18 evals per restart, that's 180. 
        # 20 left for random exploration.
        
        evals_per_restart = 18
        remaining_budget = budget - (n_restarts * evals_per_restart)
        
        # Phase 1: Local Optimization
        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if eval_count >= budget:
                break
                
            w = weights[i]
            
            # Initial point
            # Try a point biased towards the center, or random
            x0 = np.random.uniform(lo, hi)
            
            # Define scalarized objective
            # Tchebycheff: max_i (w_i * |f_i(x) - z_i^*|)
            # We don't know z^* exactly, so we use the nadir point approximation or just 0 if objectives are positive.
            # RE34 objectives are positive. Let's use a dynamic nadir or just the current best.
            # For simplicity and robustness, we can use the "augmented Tchebycheff" or just standard Tchebycheff with a dummy z*.
            # A common trick is to use the minimum observed values as z*.
            
            # However, since we are doing multi-start, we can just use the standard Tchebycheff with a fixed z* 
            # or update it. Let's use a simple scalarization: 
            # sum(w_i * f_i(x)) is simpler and often works well if objectives are of similar magnitude.
            # But RE34 objectives have different scales.
            # f1 ~ 1000s, f2 ~ 10s, f3 ~ 0.1s.
            # So we MUST normalize or use Tchebycheff.
            
            # Let's use Tchebycheff with z* estimated from a few initial points.
            # Or better, use the "weighted sum" with normalized objectives.
            # Normalization requires knowing the range. 
            # Let's do a quick 5-point sample to estimate bounds for normalization? 
            # No, that uses budget.
            
            # Alternative: Use the "PBI" (Penalty-Based Boundaries Intersection) or just Tchebycheff with z* = 0 
            # and rely on the weights to handle the scale? No, scale matters.
            
            # Let's use a simple heuristic: 
            # f1 is large, f2 medium, f3 small.
            # We can manually scale them? No, that's problem-specific.
            
            # Let's stick to Tchebycheff with z* updated dynamically during the optimization?
            # SciPy's minimize doesn't support dynamic z* easily.
            
            # Let's use a different scalarization: 
            # Minimize max( (f1 - z1)/r1, (f2 - z2)/r2, (f3 - z3)/r3 )
            # We need r (range).
            
            # Let's do a very small initial sampling (5 points) to estimate min/max for each objective.
            # This uses 5 evals.
            
            if eval_count < 5:
                # Estimate ranges
                f_samples = np.zeros((5, n_obj))
                for j in range(5):
                    x_sample = np.random.uniform(lo, hi)
                    f_samples[j] = problem(x_sample)
                    eval_count += 1
                    all_X.append(x_sample.copy())
                    all_F.append(f_samples[j].copy())
                
                f_min = np.min(f_samples, axis=0)
                f_max = np.max(f_samples, axis=0)
                f_range = f_max - f_min
                # Avoid division by zero
                f_range[f_range < 1e-6] = 1e-6
            else:
                # If we already used some, just use the current best
                f_min = np.array([0, 0, 0])
                f_range = np.array([1, 1, 1])
                
            # Define the scalarized function for this restart
            def scalarized(x, w=w, f_min=f_min, f_range=f_range):
                # Check budget
                if eval_count >= budget:
                    return np.inf
                x = np.clip(x, lo, hi)
                f = problem(x)
                eval_count += 1
                all_X.append(x.copy())
                all_F.append(f.copy())
                
                # Normalize
                f_norm = (f - f_min) / f_range
                # Tchebycheff
                val = np.max(w * f_norm)
                return val

            # Run Nelder-Mead
            try:
                res = minimize(scalarized, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_restart, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < np.inf:
                    # The best point found is in all_X/all_F
                    pass
            except Exception:
                pass

        # Phase 2: Random Exploration with Remaining Budget
        while eval_count < budget:
            if time.time() > deadline - 5:
                break
            
            x = np.random.uniform(lo, hi)
            f = problem(x)
            eval_count += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            
            # Simple local refinement around this random point?
            # If we have time, try 1 step of Nelder-Mead
            if eval_count < budget - 5 and time.time() < deadline - 10:
                def local_scalarized(x, f_min=f_min, f_range=f_range):
                    if eval_count >= budget:
                        return np.inf
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    eval_count += 1
                    all_X.append(x.copy())
                    all_F.append(f.copy())
                    # Use a random weight for diversity
                    w = np.random.dirichlet(np.ones(n_obj))
                    f_norm = (f - f_min) / f_range
                    return np.max(w * f_norm)
                
                try:
                    res = minimize(local_scalarized, x, method='Nelder-Mead',
                                   options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-4})
                except:
                    pass

        # Return the array of all evaluated points
        # The task says "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # But the seed program returns all_X. I will return all_X to be consistent.
        if len(all_X) > 0:
            return np.array(all_X)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
