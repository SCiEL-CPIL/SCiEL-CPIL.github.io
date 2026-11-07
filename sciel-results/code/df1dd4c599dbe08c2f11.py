# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Nelder-Mead simplex optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    Uses a diverse set of initial points from a coarse grid and random sampling.
    For each point, it runs a Nelder-Mead local optimization.
    This approach is designed to quickly converge to the local minimum of the 
    smooth ellipsoidal part of the function once the 'step' is crossed.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        # Generate a diverse set of starting points
        # 1. Center
        # 2. Coarse grid points (sparse)
        # 3. Random points
        
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Coarse grid: e.g., -4, 0, 4 in each dimension -> 3^5 = 243 points is too many for 500 calls
        # Let's use a smaller set. Maybe just the corners and midpoints?
        # Or a random subset of the grid.
        # Given budget 500, and NM uses ~ (dim+1) * iterations calls.
        # If we run NM 10 times, we have 50 calls each.
        # So we can afford ~10-20 starting points.
        
        # Let's generate random points and add the center.
        rng = np.random.default_rng(42)
        
        # Add some structured points
        # Corners of [-4, 4]^d? 2^5 = 32 points.
        # Let's pick a few corners.
        corners = []
        for i in range(2**dim):
            c = np.array([(-4.0 if (i >> j) & 1 else 4.0) for j in range(dim)])
            corners.append(c)
        
        # Random points
        random_points = [rng.uniform(-4.5, 4.5, dim) for _ in range(10)]
        
        # Combine and shuffle
        all_points = start_points + random_points + [corners[0], corners[-1]]
        # Shuffle to ensure diversity
        rng.shuffle(all_points)
        
        # We want to allocate budget to the best performing NM runs.
        # Strategy:
        # 1. Run a short NM from each point (e.g., 30-50 calls).
        # 2. Keep the best few.
        # 3. Run a longer NM from the best few to refine.
        
        # Phase 1: Exploration with short NM runs
        num_exploration = 10
        budget_per_exploration = 40
        
        candidates = []
        
        for i in range(num_exploration):
            if self._call_count >= budget:
                break
            if time.time() > deadline:
                break
                
            x0 = all_points[i]
            
            # Run Nelder-Mead
            # We need to wrap the function to count calls and handle time
            # scipy.optimize.minimize with method='Nelder-Mead'
            
            # We cannot easily pass a call limit to scipy.optimize.minimize directly 
            # other than maxiter. But maxiter is iterations, not function calls.
            # For Nelder-Mead, each iteration uses at most dim+1 function calls (often less if no expansion needed, but usually 1-2).
            # Let's estimate: 40 calls ~ 10-20 iterations.
            
            # To strictly control calls, we might need a custom implementation or careful maxiter.
            # However, the task says "More than 500 calls ... invalid".
            # So we must be careful.
            
            # Let's use a custom Nelder-Mead implementation to have precise control over calls.
            # Or use scipy and trust it won't blow the budget if maxiter is small.
            # maxiter=20 for dim=5 might use ~ 20 * 6 = 120 calls? No, NM is efficient.
            # Actually, standard NM in scipy uses `maxiter` as the number of iterations.
            # Each iteration involves 1 function evaluation (the new vertex) and potentially 1 reflection/expansion.
            # So ~2 calls/iteration.
            # If we set maxiter=10, that's ~20 calls.
            
            # Let's implement a simple Nelder-Mead to be safe and fast.
            
            x_best, f_best = self._nelder_mead(wrapped, x0, max_calls=30, deadline=deadline)
            
            if f_best < np.inf:
                candidates.append((f_best, x_best))

        # Phase 2: Refinement with the best candidates
        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])
        
        # Take the top 3
        for f_cand, x_cand in candidates[:3]:
            if self._call_count >= budget - 50: # Keep some buffer
                break
            if time.time() > deadline:
                break
                
            # Run a longer NM from this point
            # Use remaining budget, but cap it to ensure we don't exceed
            # We can pass a large max_calls and rely on global budget check inside
            x_best, f_best = self._nelder_mead(wrapped, x_cand, max_calls=150, deadline=deadline)
            
            if f_best < self._best_f:
                self._best_f = f_best
                self._best_x = x_best

        # Phase 3: Final Polish (Optional, if budget remains)
        if self._best_x is not None and self._call_count < budget - 10:
            # Run a very tight NM around the best point
            x_best, f_best = self._nelder_mead(wrapped, self._best_x, max_calls=budget - self._call_count - 5, deadline=deadline)
            if f_best < self._best_f:
                self._best_f = f_best
                self._best_x = x_best

        if self._best_x is None:
             self._best_x = np.zeros(dim)
             self._best_f = func(self._best_x)

        return self._best_x, self._best_f

    def _nelder_mead(self, func, x0, max_calls=100, deadline=None):
        """
        Simple Nelder-Mead implementation with call counting and time guard.
        """
        dim = len(x0)
        x = np.array(x0, dtype=float)
        
        # Initialize simplex
        # Standard simplex: x0, x0 + delta*e_i
        delta = 0.5
        simplex = [x.copy()]
        for i in range(dim):
            xi = x.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = []
        for i in range(dim + 1):
            if self._call_count >= max_calls:
                return x, np.inf
            if deadline and time.time() > deadline:
                return x, np.inf
            f = func(simplex[i])
            fvals.append(f)
        
        best_f = min(fvals)
        best_x = simplex[fvals.index(best_f)]
        
        # Main loop
        while self._call_count < max_calls:
            if deadline and time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-8:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            
            if self._call_count >= max_calls:
                break
            if deadline and time.time() > deadline:
                break
            f_r = func(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                gamma = 2.0
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                
                if self._call_count >= max_calls:
                    break
                if deadline and time.time() > deadline:
                    break
                f_e = func(x_e)
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                beta = 0.5
                x_c = centroid + beta * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                
                if self._call_count >= max_calls:
                    break
                if deadline and time.time() > deadline:
                    break
                f_c = func(x_c)
                
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        if self._call_count >= max_calls:
                            break
                        if deadline and time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
            
            # Update best
            min_val = min(fvals)
            if min_val < best_f:
                best_f = min_val
                best_x = simplex[fvals.index(min_val)]
                
        return best_x, best_f

# EVOLVE-BLOCK-END
