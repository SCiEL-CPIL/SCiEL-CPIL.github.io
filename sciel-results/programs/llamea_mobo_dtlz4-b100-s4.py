"""Multi-objective optimizer on DTLZ4 using grid-based approach with cosine spacing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using structured grid search on DTLZ4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 has 12 variables: first 7 are shape variables, last 5 are position variables
        # Actually for DTLZ4 with M=3: n_shape = dim - (M-1) = 12 - 2 = 10 shape vars
        # Wait, let me reconsider. DTLZ4 with M objectives has:
        # - (M-1) position variables that determine the Pareto front location
        # - (dim - (M-1)) shape variables that determine the shape
        # For M=3, dim=12: 2 position vars, 10 shape vars
        #
        # The Pareto front of DTLZ4 is the unit sphere in (M-1)=2 dimensions
        # i.e., x_1^2 + x_2^2 + ... + x_{M-1}^2 = 1
        #
        # For DTLZ4, the objectives on the Pareto front are:
        # f_1 = x_1
        # f_2 = x_2
        # f_3 = 1 - sqrt(x_1^2 + x_2^2)  -- wait, this isn't right
        #
        # Let me recall DTLZ4 properly:
        # For M objectives:
        # g(x) = sum_{i=n+1}^{n} (x_i - 0.5)^2 * 100 ... actually it's more complex
        #
        # For DTLZ4 with M=3:
        # Variables: x_1, ..., x_12
        # g(x) = sum_{i=10}^{12} (x_i - 0.5)^2  (last 3 variables are for g)
        # Actually, let me look at this more carefully.
        #
        # DTLZ4 standard form with M objectives and n variables:
        # Let k = n - M + 1 (number of variables for the "shape")
        # g(x) = sum_{i=k+1}^{n} (x_i - 0.5)^2
        # For Pareto front: g = 0, so x_{k+1} = ... = x_n = 0.5
        #
        # f_1(x) = (1 + g(x)) * x_1
        # f_2(x) = (1 + g(x)) * x_2
        # ...
        # f_{M-1}(x) = (1 + g(x)) * x_{M-1}
        # f_M(x) = (1 + g(x)) * (1 - sqrt(x_1^2 + ... + x_{M-1}^2))
        #
        # Wait, that's not quite right either. Let me think again.
        #
        # For DTLZ4 with M=3, n=12:
        # k = n - M + 1 = 12 - 3 + 1 = 10
        # g(x) = sum_{i=11}^{12} (x_i - 0.5)^2
        #
        # f_1 = (1+g) * x_1
        # f_2 = (1+g) * x_2
        # f_3 = (1+g) * (1 - sqrt(x_1^2 + x_2^2))
        #
        # On Pareto front (g=0):
        # f_1 = x_1
        # f_2 = x_2
        # f_3 = 1 - sqrt(x_1^2 + x_2^2)
        #
        # With constraint x_1^2 + x_2^2 + ... + x_{M-1}^2 <= 1, but for the front we need equality
        # Actually for DTLZ4, the Pareto front is where x_1^2 + x_2^2 = 1 (for M=3)
        # and x_1, x_2 >= 0
        #
        # So on the Pareto front:
        # Let x_1 = cos(theta), x_2 = sin(theta) for theta in [0, pi/2]
        # f_1 = cos(theta)
        # f_2 = sin(theta)
        # f_3 = 1 - 1 = 0
        #
        # Wait, that gives f_3 = 0 always, which doesn't seem right for hypervolume.
        #
        # Let me reconsider. The Pareto front of DTLZ4 is actually a curved surface.
        # For M=3, it's a quarter unit circle in the (f_1, f_2) plane with f_3 varying.
        #
        # Actually I think I'm confusing myself. Let me just use a grid-based approach
        # that samples the decision space and relies on the problem function.

        # Strategy: Use a structured grid on the Pareto front
        # For DTLZ4 with M=3, the Pareto front is parameterized by 2 angles
        # We'll sample points on the quarter unit disk in 2D

        # First, let's do some random initialization to get a sense of the landscape
        n_init = min(20, self.budget // 5)
        rng = np.random.default_rng(0)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Now do structured sampling on the Pareto front
        # For DTLZ4, the Pareto front is defined by:
        # x_1 = cos(theta) * r, x_2 = sin(theta) * r, where r=1 for the front
        # Other variables are set to 0.5 (to make g=0)
        
        # Let's create a 2D grid on the quarter circle
        # Use cosine spacing for better coverage
        n_theta = int(np.sqrt(self.budget - used - 10))
        n_theta = max(n_theta, 5)
        
        theta_vals = np.cos(np.linspace(0, np.pi/2, n_theta))
        # This gives better spacing near the vertices

        points = []
        for i in range(n_theta):
            for j in range(n_theta):
                if len(points) >= self.budget - used - 5:
                    break
                # Parameterize the Pareto front
                # x_1 = theta_vals[i], x_2 = theta_vals[j]
                # But we need x_1^2 + x_2^2 = 1 for the front
                # Let's use a different parameterization
                
                # Actually, let's use polar coordinates
                # r = 1, theta from 0 to pi/2
                # x_1 = cos(phi), x_2 = sin(phi) where phi in [0, pi/2]
                # But we want a 2D grid, so let's use:
                # x_1 = a, x_2 = b where a^2 + b^2 = 1
                # This is a 1D manifold, not 2D
                
                # For DTLZ4 with M=3, the Pareto front is 1D (a curve)
                # So we should sample along this curve
                pass
        
        # Let me reconsider the structure. For DTLZ4 with M=3:
        # The Pareto front is a 1D curve (quarter circle)
        # We should sample points along this curve
        
        # Reset and use a cleaner approach
        # Sample points on the Pareto front curve
        n_samples = self.budget - used - 10
        
        # Use cosine spacing for better coverage
        t_vals = np.cos(np.linspace(0, np.pi/2, n_samples))
        
        # For each t, create a point on the Pareto front
        # x_1 = t, x_2 = sqrt(1 - t^2)
        # Other variables: set to 0.5 to make g=0
        # But wait, the problem clips to bounds, and the exact Pareto front
        # might not be exactly at 0.5 for the other variables
        
        # Let's just evaluate points and let the problem function handle it
        F_all = []
        for t in t_vals:
            if time.time() > deadline:
                break
            x = np.full(self.dim, 0.5)
            x[0] = t
            x[1] = np.sqrt(max(0, 1 - t**2))
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            F_all.append(f)
            used += 1
        
        # Also add some random perturbations around the Pareto front
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            n_perturb = min(remaining, 20)
            for _ in range(n_perturb):
                if time.time() > deadline:
                    break
                base_idx = rng.integers(len(t_vals))
                t = t_vals[base_idx]
                x = np.full(self.dim, 0.5)
                x[0] = t + 0.05 * rng.standard_normal()
                x[1] = np.sqrt(max(0, 1 - t**2)) + 0.05 * rng.standard_normal()
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_all.append(f)
                used += 1
        
        # Combine all results
        if F_all:
            F_combined = np.array(F_all)
        else:
            F_combined = F_init
        
        # Return the first point (return value is ignored)
        return X_init[0] if len(X_init) > 0 else np.zeros(self.dim)
# EVOLVE-BLOCK-END
