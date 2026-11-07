import numpy as np


class Optimizer:
    """Adaptive lattice-based random walk with sparse directions for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Pre-generate a set of sparse random directions
        # We want directions that move along 1, 2, or 3 axes simultaneously
        # to better navigate rotated landscapes
        n_dirs = 200
        directions = []
        for _ in range(n_dirs):
            d = np.zeros(dim)
            # Pick 1 to min(3, dim) axes
            n_axes = rng.integers(1, min(4, dim) + 1)
            axes = rng.choice(dim, size=n_axes, replace=False)
            for a in axes:
                d[a] = rng.choice([-1.0, 1.0])
            norm = np.linalg.norm(d)
            if norm > 0:
                directions.append(d / norm)
        directions = np.array(directions)

        # Initial best
        x_best = rng.uniform(-4, 4, dim)
        fx_best = func(x_best)
        calls = 1

        # Current state
        x_curr = x_best.copy()
        fx_curr = fx_best
        h = 1.0  # Initial step size, commensurate with Rastrigin period
        
        # Parameters for adaptation
        h_min = 1e-6
        h_max = 2.0
        shrink_factor = 0.5
        grow_factor = 1.1 # Slower growth to avoid overshooting
        max_stuck = 15
        stuck = 0

        while calls < budget:
            # Generate a candidate move
            d = directions[rng.integers(n_dirs)]
            # Jitter the step size slightly to avoid cycling
            h_eff = h * rng.uniform(0.9, 1.1)
            
            y = x_curr + h_eff * d
            # Clip to bounds
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1

            if fy < fx_curr:
                # Accept move
                x_curr = y
                fx_curr = fy
                stuck = 0
                
                # If we found a new global best, update it
                if fy < fx_best:
                    x_best = y.copy()
                    fx_best = fy
                    
                # Grow step size slightly to explore further, but cap it
                # This helps escape shallow local minima
                if h < h_max:
                    h = min(h * grow_factor, h_max)
            else:
                # Reject move
                stuck += 1
                
                # If stuck, shrink step size to refine
                if stuck >= max_stuck:
                    h = max(h * shrink_factor, h_min)
                    stuck = 0
                    
                    # If step size is extremely small, we are likely in a local minimum
                    # Try a random restart or a large jump to escape
                    if h <= h_min * 1.5:
                        # Option 1: Global restart
                        if rng.random() < 0.2:
                            x_curr = rng.uniform(-4, 4, dim)
                            fx_curr = func(x_curr)
                            calls += 1
                            if fx_curr < fx_best:
                                x_best = x_curr.copy()
                                fx_best = fx_curr
                            h = 1.0
                        else:
                            # Option 2: Local jump (perturb current best)
                            # Jump to a random point in a box of size 2 around best
                            x_curr = x_best + rng.uniform(-2, 2, dim)
                            x_curr = np.clip(x_curr, -5, 5)
                            fx_curr = func(x_curr)
                            calls += 1
                            if fx_curr < fx_best:
                                x_best = x_curr.copy()
                                fx_best = fx_curr
                            h = 0.5
                            stuck = 0

        return x_best, fx_best
