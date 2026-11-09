import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def _halton(self, index, base):
        """Generate the index-th number in the Halton sequence for the given base."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_sequence(self, n, dim):
        """Generate n points in dim dimensions using the Halton sequence."""
        points = np.zeros((n, dim))
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
        for i in range(n):
            for j in range(dim):
                # Skip the first point (0) to avoid boundary issues, or map [0,1) to [-5,5]
                # Halton usually starts at 1 for the first index if we want (0,1)
                # Standard Halton: k-th point (1-indexed)
                points[i, j] = self._halton(i + 1, bases[j])
        # Map [0, 1) to [-5, 5]
        points = points * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return points

    def _nelder_mead(self, func, x0, budget, deadline):
        """Run Nelder-Mead optimization starting from x0."""
        dim = self.dim
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += 0.1 # Small perturbation
        
        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        calls = 0
        for i in range(dim + 1):
            if calls >= budget or time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls += 1
        
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        max_iter = budget
        iter_count = 0
        
        while iter_count < max_iter and calls < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check convergence
            if np.std(fvals) < 1e-10:
                break
            if np.max(np.abs(simplex - simplex[0])) < 1e-8:
                break

            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, self.bounds[0], self.bounds[1])
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, self.bounds[0], self.bounds[1])
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                # Outside contraction if f_ref < fvals[-1], inside otherwise
                if f_ref < fvals[-1]:
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    x_con = centroid - rho * (centroid - simplex[-1])
                x_con = np.clip(x_con, self.bounds[0], self.bounds[1])
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_ref:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls >= budget or time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls += 1
                    
            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()
            
            iter_count += 1
            
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin for 300s limit
        
        # 1. Global Sampling
        n_global = 20
        global_points = self._halton_sequence(n_global, self.dim)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Evaluate global points
        for x in global_points:
            if time.time() > deadline:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # 2. Local Refinement with Multi-Start Nelder-Mead
        # Select top 5 points from global sampling to refine
        # We need to track which points were evaluated
        # For simplicity, we'll just re-evaluate the best few or pick diverse ones.
        # A better approach: keep the top K unique points.
        
        # Let's pick the best 5 points from the global set to start NM.
        # We need their values. We didn't store them all. 
        # Let's modify the loop to store them.
        
        # Re-do global sampling with storage
        global_results = []
        calls = 0
        best_x = None
        best_f = float('inf')
        
        for x in global_points:
            if time.time() > deadline:
                break
            fx = func(x)
            calls += 1
            global_results.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort by function value
        global_results.sort(key=lambda t: t[0])
        
        # Take top 5 starts
        n_starts = 5
        start_points = [x for _, x in global_results[:n_starts]]
        
        # Allocate remaining budget
        remaining_budget = self.budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Distribute budget among starts
        # Give more budget to better starts?
        # Simple equal split:
        budget_per_start = remaining_budget // n_starts
        
        for i, x_start in enumerate(start_points):
            if time.time() > deadline:
                break
            
            # Allocate slightly more to the first (best) start
            if i == 0:
                b = budget_per_start + (remaining_budget % n_starts)
            else:
                b = budget_per_start
            
            if b <= 0:
                continue
                
            try:
                x_res, f_res, calls_used = self._nelder_mead(func, x_start, b, deadline)
                calls += calls_used
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res
            except Exception:
                pass

        return best_x, best_f
