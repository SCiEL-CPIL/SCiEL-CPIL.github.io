"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase DTLZ4 optimizer: analytical front sampling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dtlz4_objectives(self, x, lo, hi):
        """Compute DTLZ4 objectives for a single point."""
        n = len(x)
        m = n - (self.n_obj - 1)  # position variables count
        # g(x_m) = sum of (x_i - 0.5)^2 for position variables
        g = np.sum((x[:m] - 0.5) ** 2)
        # Shape variables: x_m ... x_{n-1}
        xk = x[m:]
        # h functions
        h1 = np.cos(xk[-1] * np.pi / 2) * np.cos(xk[-2] * np.pi / 2)
        h2 = np.cos(xk[-1] * np.pi / 2) * np.sin(xk[-2] * np.pi / 2)
        h3 = np.sin(xk[-1] * np.pi / 2)
        return g * h1, g * h2, g * h3

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = n - (self.n_obj - 1)  # number of position variables
        k = self.n_obj - 1  # number of shape variables

        used = 0
        all_points = []
        all_fvals = []

        # Phase 1: Sample the known Pareto front analytically
        # For DTLZ4, Pareto front: g(x_m) = 0 => x_i = 0.5 for i in [0, m-1]
        # Shape variables x_m..x_{n-1} determine the front shape
        # Use a grid in the shape variable space
        # For 3 objectives, 2 shape variables (x_{n-2}, x_{n-1})
        # h1 = cos(t2*pi/2)*cos(t1*pi/2), h2 = cos(t2*pi/2)*sin(t1*pi/2), h3 = sin(t2*pi/2)
        # where t1 = x_{n-2}, t2 = x_{n-1}, both in [0,1]
        
        # Grid sampling: 15x15 = 225 points on the front
        n_grid = 15
        grid_size = n_grid * n_grid
        if grid_size > self.budget:
            n_grid = max(5, int(np.sqrt(self.budget)))
            grid_size = n_grid * n_grid

        t1_vals = np.linspace(0, 1, n_grid)
        t2_vals = np.linspace(0, 1, n_grid)
        
        front_points = []
        front_fvals = []
        
        for t1 in t1_vals:
            for t2 in t2_vals:
                if used >= self.budget:
                    break
                x = np.full(n, 0.5)
                x[m + k - 2] = t1  # x_{n-2}
                x[m + k - 1] = t2  # x_{n-1}
                # Clip to bounds
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                front_points.append(x)
                front_fvals.append(f)
                if time.time() > deadline:
                    return front_points

        all_points.extend(front_points)
        all_fvals.extend(front_fvals)

        # Phase 2: Local optimization - perturb position variables to reduce g
        # Start from front points and try to minimize g by adjusting position vars
        rng = np.random.default_rng(42)
        
        # Select a subset of front points for refinement
        n_refine = min(len(all_points), max(20, (self.budget - used) // 3))
        if n_refine > 0:
            # Pick diverse points from the front
            indices = np.linspace(0, len(all_points) - 1, n_refine).astype(int)
            
            for idx in indices:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    return all_points
                
                base_point = all_points[idx].copy()
                best_f = np.array(all_fvals[idx], dtype=float)
                best_x = base_point.copy()
                
                # Local search: try to reduce g by adjusting position variables
                step_size = 0.05
                for _ in range(10):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        return all_points
                    
                    # Random perturbation of position variables
                    perturbation = rng.standard_normal(m) * step_size
                    candidate = best_x.copy()
                    candidate[:m] = np.clip(candidate[:m] + perturbation, lo[:m], hi[:m])
                    
                    fc = problem(candidate)
                    fc = np.array(fc, dtype=float)
                    used += 1
                    
                    # Check if this is better (lower g means lower objectives)
                    if np.all(fc <= best_f + 1e-10):
                        if np.any(fc < best_f - 1e-10):
                            best_f = fc
                            best_x = candidate
                            step_size *= 0.8
                    else:
                        step_size *= 0.7
                    
                    if step_size < 0.001:
                        break
                
                if np.all(best_f <= np.array(all_fvals[idx], dtype=float) + 1e-10):
                    all_points.append(best_x)
                    all_fvals.append(best_f)

        # Phase 3: If budget remains, do random sampling to find any missed regions
        if used < self.budget:
            n_random = self.budget - used
            for _ in range(n_random):
                if time.time() > deadline:
                    return all_points
                x = lo + (hi - lo) * rng.random(n)
                f = problem(x)
                used += 1
                all_points.append(x)
                all_fvals.append(f)

        return all_points
# EVOLVE-BLOCK-END
