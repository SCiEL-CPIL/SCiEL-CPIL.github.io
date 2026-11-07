"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based global exploration followed by adaptive local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin before 300s limit
        calls = 0

        def can_continue():
            return calls < budget and time.time() < deadline

        def remaining():
            return budget - calls

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # Phase 1: Systematic Grid Exploration
        # For dim=5, a 3^5=243 point grid is affordable and provides global coverage.
        # This ensures we find the correct basin early, boosting AOCC.
        if dim == 5:
            # 3 levels: -3, 0, 3. This covers [-3,3] which contains the minimizer [-4,4].
            # Actually, to be safe, let's use -4, 0, 4.
            levels = np.array([-4.0, 0.0, 4.0])
        else:
            # Fallback for other dims (though task says dim=5)
            levels = np.linspace(-5, 5, 3)

        # Generate all grid points
        grid_points = []
        for l1 in levels:
            for l2 in levels:
                for l3 in levels:
                    for l4 in levels:
                        for l5 in levels:
                            grid_points.append(np.array([l1, l2, l3, l4, l5]))

        # Add center and a few random points to hedge against grid alignment issues
        grid_points.append(np.zeros(dim))
        
        # Shuffle to avoid systematic bias in early calls? No, grid order is fine, 
        # but maybe center first is better for AOCC if origin is often good?
        # Actually, random order is safer to not waste budget on a bad corner if origin is good.
        # But we want systematic coverage. Let's just evaluate them.
        
        # To maximize early AOCC, it's often better to start with a "reasonable" guess 
        # like the origin, then expand.
        # Let's put origin first.
        if dim == 5:
             grid_points = [np.zeros(5)] + [p for p in grid_points if not np.allclose(p, 0)]

        # Evaluate grid
        for x in grid_points:
            if not can_continue():
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            update_best(x, fx)

        if not can_continue():
            return best_x, best_f

        # Phase 2: Adaptive Local Refinement (1/5-rule variant)
        # Start from the best point found in Phase 1.
        # Use a coordinate-wise random search with adaptive step size.
        # This is robust and fast, avoiding the overhead of CMA-ES matrix ops.
        
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        # We will use a simple random walk with step adaptation.
        # If a step improves, increase sigma slightly (or keep). If not, decrease.
        # Standard 1/5 rule: increase step if success rate > 0.2, else decrease.
        # For simplicity and speed, we'll do a greedy hill-climbing with random perturbations.
        
        rng = np.random.default_rng(42)
        
        # To make it more effective, we can use a "ladder" approach:
        # Try different step sizes.
        # Or just a standard adaptive random search.
        
        # Let's use a simple adaptive random search:
        # 1. Generate random direction
        # 2. Try step size sigma
        # 3. If better, accept, maybe increase sigma
        # 4. If worse, reject, decrease sigma
        
        # To improve convergence, we can also try coordinate-wise steps.
        # But full-dimensional random steps are often fine for low dim.
        
        # Let's use a hybrid: mostly full-dim, sometimes coordinate-wise.
        
        while can_continue():
            # Generate random direction
            if rng.random() < 0.5:
                # Full dimensional
                direction = rng.standard_normal(dim)
            else:
                # Coordinate-wise
                direction = np.zeros(dim)
                i = rng.integers(0, dim)
                direction[i] = rng.choice([-1.0, 1.0])
            
            # Normalize to unit length to make sigma meaningful
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            
            x_new = x + sigma * direction
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1
            
            if f_new < fx:
                x = x_new
                fx = f_new
                update_best(x, fx)
                # Increase step size slightly to explore faster
                sigma *= 1.05
                if sigma > 5.0:
                    sigma = 5.0
            else:
                # Decrease step size to refine
                sigma *= 0.95
                if sigma < 1e-4:
                    # If step is too small, try a random restart around current best?
                    # Or just break?
                    # Let's try a few more times with a slightly larger step to escape local min
                    sigma = 0.1
                    # If we are stuck, maybe do a small random jump
                    if rng.random() < 0.1:
                        jump = rng.standard_normal(dim) * 0.5
                        x_try = x + jump
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < fx:
                            x = x_try
                            fx = f_try
                            update_best(x, fx)
                            sigma = 0.1
                        else:
                            sigma = 0.01

        return best_x, best_f

# EVOLVE-BLOCK-END
