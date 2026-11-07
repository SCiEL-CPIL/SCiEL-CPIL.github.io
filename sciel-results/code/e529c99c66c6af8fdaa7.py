"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-restart local search with composite objective minimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for internal RNG to be deterministic if needed, 
        # but the evaluator seeds np.random. We use default_rng for our own control.
        rng = np.random.default_rng(42)
        
        # Archive of all evaluated points
        X_archive = []
        F_archive = []
        
        used = 0
        
        # Helper to add a point to archive
        def add_point(x, f):
            nonlocal used
            X_archive.append(x)
            F_archive.append(f)
            used += 1

        # 1. Initial Random Sampling (10% of budget)
        n_init = max(10, budget // 10)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        for x in X_init:
            if used >= budget: break
            if time.time() > deadline: break
            f = problem(x)
            add_point(x, f)
        
        if used >= budget or time.time() > deadline:
            return np.array(X_archive)

        # 2. Main Loop: Local Search from best points
        # We will pick the current "best" point (lowest sum of objectives, or non-dominated)
        # and run a local minimization on a weighted sum.
        
        # Weights for composite objective. We'll cycle through different weight vectors
        # to cover different parts of the Pareto front.
        # Since we want to minimize all 3, and reference is [1705.0, 12.0, 0.3],
        # the scales are very different. We should normalize or use adaptive weights.
        # Let's try a few static weight sets that emphasize different objectives.
        # Obj 1 is ~1000s, Obj 2 is ~10s, Obj 3 is ~1s.
        # To balance, we might weight Obj 3 higher.
        
        weight_sets = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 10.0, 100.0]), # Emphasize Obj 3
            np.array([1.0, 1.0, 10.0]),   # Emphasize Obj 2 and 3
            np.array([10.0, 1.0, 1.0]),   # Emphasize Obj 1
            np.array([1.0, 10.0, 1.0]),   # Emphasize Obj 2
            np.array([1.0, 1.0, 1.0]),    # Balanced again
        ]
        
        w_idx = 0
        n_restarts = 0
        
        while used < budget and time.time() < deadline:
            # Select a starting point for local search
            # Strategy: Pick a point from the archive that is non-dominated or has low composite value
            if len(F_archive) == 0:
                break
            
            F_arr = np.array(F_archive)
            X_arr = np.array(X_archive)
            
            # Compute composite score using current weights
            w = weight_sets[w_idx % len(weight_sets)]
            scores = F_arr @ w
            best_idx = np.argmin(scores)
            x0 = X_arr[best_idx]
            f0 = F_arr[best_idx]
            
            # Define composite objective function for scipy
            def composite_func(x):
                x = np.clip(x, lo, hi)
                f = problem(x)
                nonlocal used
                used += 1
                # Record this point? 
                # Recording every intermediate step might be good for hypervolume, 
                # but scipy does many internal evals. 
                # Let's record them.
                X_archive.append(x)
                F_archive.append(f)
                return float(f @ w)
            
            # Run Nelder-Mead
            # maxiter should be small to allow many restarts
            # We have ~180 evaluations left after init.
            # If we do 10 restarts, we have ~18 evals per restart.
            # Nelder-Mead in 5D might need more to converge, but it's fast.
            
            res = minimize(
                composite_func, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            # Check if we exceeded budget or time during minimize
            if used >= budget or time.time() > deadline:
                break
                
            # Cycle to next weight set for next restart
            w_idx += 1
            n_restarts += 1
            
            # Occasionally, add a random point to explore new areas
            if n_restarts % 3 == 0 and used < budget - 10:
                x_rand = lo + (hi - lo) * rng.random(dim)
                f_rand = problem(x_rand)
                add_point(x_rand, f_rand)

        return np.array(X_archive)
# EVOLVE-BLOCK-END
