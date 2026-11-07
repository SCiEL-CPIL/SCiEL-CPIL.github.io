# EVOLVE-BLOCK-START
import numpy as np
import time
import random

class Optimizer:
    """
    Multi-start CMA-ES Optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    
    Strategy:
    1. Perform a coarse global search using multiple short CMA-ES runs.
    2. Each run starts from a different random center in the box.
    3. Use a dynamic budget allocation: give more budget to runs that find better minima.
    4. The "Step" nature of f7 is handled by CMA-ES's ability to adapt its covariance
       matrix to the local ellipsoidal structure once it enters a basin.
    5. Time-guarded execution to ensure completion within limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin (300s limit)
        dim = self.dim
        budget = self.budget
        
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None
        
        rng = np.random.default_rng(42)
        
        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        # CMA-ES Implementation
        def run_cmaes(center, max_evals, sigma0):
            """
            Run CMA-ES starting from center with a specific budget.
            Returns the best point and function value found in this run.
            """
            if max_evals <= 0 or self._call_count >= budget or time.time() > deadline:
                return center, np.inf
            
            # Population size
            lam = 4 + int(3 * np.log(dim))
            
            # Initial state
            mean = np.array(center, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # CMA-ES constants
            cmu = 2 / ((dim + 4) ** 2 + lam)
            cs = (lam_cmu + 2) / (dim + lam_cmu + 5)
            ds = 1 + 2 * max(0, np.sqrt((lam_cmu - 1) / (dim + 1)) - 1) + cs
            cc = (4 + lam_cmu / dim) / (dim + 4 + 2 * lam_cmu / dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            # Inverse square root of C
            def invsqrt(A):
                # Use eigendecomposition for stability
                w, v = np.linalg.eigh(A)
                w = np.maximum(w, 1e-10)
                return v @ np.diag(1 / np.sqrt(w)) @ v.T
            
            Cs = invsqrt(C)
            
            best_x_run = mean.copy()
            best_f_run = np.inf
            
            # Pre-generate random numbers for efficiency? No, sequential is better for adaptivity.
            
            while self._call_count < budget and (budget - self._call_count) >= max_evals and time.time() < deadline:
                # Actually, we want to stop if we exceed the specific run budget
                # Let's track local evals
                local_evals = 0
                while local_evals < max_evals and self._call_count < budget and time.time() < deadline:
                    # Generate population
                    ys = []
                    xs = []
                    for i in range(lam):
                        # Sample from N(0, C)
                        # y ~ N(0, I), x = mean + sigma * C^0.5 * y
                        y = rng.normal(0, 1, dim)
                        x = mean + sigma * (Cs @ y)
                        # Clip to box
                        x = np.clip(x, -5.0, 5.0)
                        
                        # Evaluate
                        fx = wrapped(x)
                        ys.append(y)
                        xs.append((x, fx))
                        local_evals += 1
                        
                        if fx < best_f_run:
                            best_f_run = fx
                            best_x_run = x.copy()
                    
                    # Sort by function value
                    xs.sort(key=lambda t: t[1])
                    
                    # Update mean
                    mu = int(lam / 2)
                    weights = np.log(lam / 2 + 0.5) - np.log(1 + np.arange(mu))
                    weights /= weights.sum()
                    
                    # Update CMA-ES state
                    y_new = np.array([xs[i][0] for i in range(mu)])
                    mean_old = mean.copy()
                    mean = mean + sigma * (Cs @ (weights @ y_new))
                    
                    # Update covariance matrix
                    ys_sorted = np.array([ys[i] for i in range(mu)])
                    delta_mean = (mean - mean_old) / sigma
                    C = (1 - cmu * cs * (1 - cc)) * C + cmu * (np.outer(delta_mean, delta_mean) + (1 - cs * (1 - cc)) / dim * np.diag(np.sum(ys_sorted**2, axis=0)))
                    
                    # Update step size (1/5 power rule)
                    ps = (1 - cs) * ps + cs / np.sqrt(cs * (2 - cs) * (1 - cc)) * (delta_mean @ Cs)
                    sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                    
                    # Update C^0.5
                    Cs = invsqrt(C)
                    
                    # Check for convergence/stagnation
                    if np.linalg.norm(mean - mean_old) < 1e-12 * sigma:
                        break
                        
                    if time.time() > deadline:
                        break

            return best_x_run, best_f_run

        # Strategy: Multi-start with dynamic allocation
        # Phase 1: Coarse search
        # Use a few diverse starting points
        
        # Determine number of restarts based on budget
        # If budget is large, we can afford more diverse starts
        n_restarts = 5
        # Base budget per restart
        # Reserve some for final refinement
        
        # List of centers
        centers = []
        # Center 0: Origin (often good default)
        centers.append(np.zeros(dim))
        # Other centers: Random
        for _ in range(n_restarts - 1):
            centers.append(rng.uniform(-4.0, 4.0, dim))
        
        # Shuffle centers to randomize order? No, keep origin first for quick baseline.
        
        # Dynamic allocation:
        # Start each with a small chunk, expand the best.
        
        # Initial chunk size
        chunk = max(10, budget // (n_restarts * 2))
        
        # Run initial chunks
        best_centers = []
        best_vals = []
        
        for i, c in enumerate(centers):
            if self._call_count >= budget or time.time() > deadline:
                break
            # Run a short CMA-ES
            x_best, f_best = run_cmaes(c, chunk, 2.0)
            best_centers.append(x_best)
            best_vals.append(f_best)
            
            # Update global best
            if f_best < self._best_f:
                self._best_f = f_best
                self._best_x = x_best

        # Refinement:
        # Take the best center from Phase 1 and run a longer CMA-ES from there.
        # Also take the second best and run a shorter one.
        
        if self._call_count < budget and time.time() < deadline:
            # Sort centers by their local best value
            sorted_indices = np.argsort(best_vals)
            
            # Allocate remaining budget to top performers
            # Top 1 gets 50% of remaining, Top 2 gets 25%, etc.
            remaining = budget - self._call_count
            
            # Top 1
            idx1 = sorted_indices[0]
            alloc1 = int(remaining * 0.6)
            if alloc1 > 0:
                x_best, f_best = run_cmaes(best_centers[idx1], alloc1, 1.0)
                if f_best < self._best_f:
                    self._best_f = f_best
                    self._best_x = x_best
                    
            if self._call_count < budget and time.time() < deadline:
                # Top 2
                idx2 = sorted_indices[1]
                remaining2 = budget - self._call_count
                alloc2 = int(remaining2 * 0.4)
                if alloc2 > 0:
                    x_best, f_best = run_cmaes(best_centers[idx2], alloc2, 1.0)
                    if f_best < self._best_f:
                        self._best_f = f_best
                        self._best_x = x_best
                        
            # If there is still budget, do a final polish on the global best
            if self._call_count < budget and time.time() < deadline:
                remaining3 = budget - self._call_count
                alloc3 = remaining3
                if alloc3 > 0:
                    x_best, f_best = run_cmaes(self._best_x, alloc3, 0.5)
                    if f_best < self._best_f:
                        self._best_f = f_best
                        self._best_x = x_best

        return self._best_x, self._best_f

# EVOLVE-BLOCK-END
