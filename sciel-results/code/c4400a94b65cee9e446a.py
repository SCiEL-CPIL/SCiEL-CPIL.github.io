"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using scalarized local search (L-BFGS-B) on RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f_new, archive):
        """Check if f_new is dominated by any point in archive."""
        for f in archive:
            if np.all(f <= f_new) and np.any(f < f_new):
                return True
        return False

    def _update_archive(self, f_new, archive, bounds, ref):
        """
        Update the non-dominated archive with a new point.
        Returns True if the point was added, False otherwise.
        Also removes points dominated by f_new.
        """
        # Check if dominated
        if self._is_dominated(f_new, archive):
            return False
        
        # Remove points dominated by f_new
        new_archive = []
        for f in archive:
            if not (np.all(f_new <= f) and np.any(f_new < f)):
                new_archive.append(f)
        
        new_archive.append(f_new)
        return True

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Archive stores objective vectors
        archive = []
        
        # We will track evaluations
        eval_count = 0
        
        # Helper to evaluate and add to archive if non-dominated
        def evaluate_and_add(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            
            # Only add if it is better than reference in all objectives
            if np.all(f < ref):
                self._update_archive(f, archive, bounds, ref)
            return f

        # 1. Initial Random Sampling to find diverse starting points
        # Use a Sobol-like or simple random spread
        n_init = 20
        rng = np.random.default_rng(0)
        
        # Generate initial points
        # We want to cover the space well. 
        # Simple random is fine for 5D with 20 points.
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            evaluate_and_add(x)
            
            if eval_count >= budget or time.time() > deadline:
                break

        # 2. Local Refinement using L-BFGS-B with different weight vectors
        # We will perform scalarized optimizations.
        # For 3 objectives, we can use weighted sum: min sum(w_i * f_i)
        # Or Tchebycheff: min max(w_i * (f_i - z_i))
        
        # Let's define a set of weight vectors to explore the front.
        # Since we have a limited budget, we should be strategic.
        # Each L-BFGS-B run might take 10-50 evaluations depending on convergence.
        # We have ~180 evals left.
        # Let's try ~15-20 local optimizations.
        
        # Generate weight vectors on the simplex for 3 objectives
        # We can use a grid or random weights.
        # Let's use a set of predefined weights that cover the front well.
        weights = []
        # Add vertices
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        
        # Add midpoints
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        
        # Add center
        weights.append([1/3, 1/3, 1/3])
        
        # Add some random weights for diversity
        for _ in range(10):
            w = rng.random(3)
            w /= np.sum(w)
            weights.append(w)
        
        # Shuffle weights to mix up the search order
        rng.shuffle(weights)
        
        # For each weight vector, perform a local search
        # We need starting points. We can use the current archive points as starting points.
        # Or random points. Using archive points is better as they are already good.
        
        # If archive is empty, we have a problem, but we did 20 random inits, so it shouldn't be.
        
        for w in weights:
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Select a starting point
            # Pick a random point from the archive if available, else random
            if len(archive) > 0:
                # Pick a random archive point
                start_f = archive[rng.integers(len(archive))]
                # We don't have the x corresponding to start_f directly in this simple structure.
                # We need to keep track of x as well.
                # Let's refactor slightly: keep a list of (x, f) pairs for the archive?
                # Or just pick a random x in bounds?
                # Picking a random x is safe but less efficient.
                # Let's keep a simple list of x's that led to archive points.
                # Actually, let's just restart the search from random points or the best found so far.
                # To keep it simple and robust, let's just use random starting points for local search
                # but constrained to be near the "center" of the bounds or random.
                
                # Better: Keep a list of x's for the archive.
                # Let's modify the structure to store (x, f) in the archive.
                pass
            
            # Refactored Archive: List of [x, f]
            # Let's restart the logic with a cleaner archive structure.
            
            break # Break out to implement the cleaner version below

        # RE-IMPLEMENTATION with cleaner state management
        
        # Reset
        archive_xf = [] # List of (x, f) tuples
        eval_count = 0
        
        def evaluate_and_add_clean(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            
            if np.all(f < ref):
                # Check domination
                dominated = False
                for _, f_old in archive_xf:
                    if np.all(f_old <= f) and np.any(f_old < f):
                        dominated = True
                        break
                
                if not dominated:
                    # Remove dominated points
                    new_archive_xf = []
                    for x_old, f_old in archive_xf:
                        if not (np.all(f <= f_old) and np.any(f < f_old)):
                            new_archive_xf.append((x_old, f_old))
                    
                    new_archive_xf.append((x, f))
                    archive_xf[:] = new_archive_xf
            return f

        # 1. Initial Random Sampling
        for _ in range(n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate_and_add_clean(x)

        # 2. Local Search Phase
        # We will iterate through weight vectors.
        # For each weight vector, we pick a random starting point from the current archive
        # (or a random point if archive is small) and run L-BFGS-B.
        
        # We need to be careful with the number of evaluations per local search.
        # L-BFGS-B maxiter can be set to limit evaluations.
        # However, problem(x) is called inside the objective function.
        # We need to count those calls.
        
        # Let's create a wrapper for the objective function that counts evaluations.
        # But we already have `eval_count` in the closure.
        
        def objective_scalar(x, w):
            if eval_count >= budget:
                return 1e10
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            # Weighted sum
            return np.dot(w, f)

        # We need to ensure we don't exceed budget inside the minimizer.
        # The minimizer might call the function many times.
        # We can set maxiter to a small number, e.g., 10-20.
        # But maxiter is not strictly the number of function evaluations, though it's correlated.
        # A safer way is to check `eval_count` inside the objective and return a large value if exceeded.
        
        for w in weights:
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Choose starting point
            if len(archive_xf) > 0:
                # Pick a random point from archive
                idx = rng.integers(len(archive_xf))
                x0 = archive_xf[idx][0].copy()
            else:
                x0 = lo + (hi - lo) * rng.random(dim)
            
            # Add some noise to x0 to escape local minima
            x0 += 0.1 * (hi - lo) * rng.standard_normal(dim)
            x0 = np.clip(x0, lo, hi)
            
            try:
                res = minimize(
                    objective_scalar, 
                    x0, 
                    args=(w,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 15, 'ftol': 1e-6}
                )
                
                # Evaluate the final point to ensure it's in the archive
                # The minimizer might have stopped due to budget, so res.x might be the last valid one
                # Or it might be a point that wasn't evaluated if it failed.
                # Let's explicitly evaluate res.x
                if eval_count < budget and time.time() < deadline:
                    evaluate_and_add_clean(res.x)
                    
            except Exception:
                pass

        # 3. Final Polish: If we have budget left, do more random local searches
        # Pick random points from archive and do very short local searches with random weights
        
        while eval_count < budget and time.time() < deadline:
            # Pick a random weight
            w = rng.random(3)
            w /= np.sum(w)
            
            if len(archive_xf) > 0:
                idx = rng.integers(len(archive_xf))
                x0 = archive_xf[idx][0].copy()
            else:
                x0 = lo + (hi - lo) * rng.random(dim)
            
            x0 += 0.05 * (hi - lo) * rng.standard_normal(dim)
            x0 = np.clip(x0, lo, hi)
            
            try:
                res = minimize(
                    objective_scalar, 
                    x0, 
                    args=(w,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 5, 'ftol': 1e-5}
                )
                if eval_count < budget and time.time() < deadline:
                    evaluate_and_add_clean(res.x)
            except Exception:
                pass

        return None
# EVOLVE-BLOCK-END
