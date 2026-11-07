import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        x_best = None
        fx_best = float('inf')
        calls = 0

        def track(x, fx):
            nonlocal x_best, fx_best
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()

        def safe_call(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            track(x, fx)
            return fx

        # Phase 1: Lattice-based exploration to exploit Rastrigin's periodic structure
        # Rastrigin has minima at integer points (in its native frame).
        # We sample on a grid with step 1.0 to hit near-minima.
        n_lattice = min(40, budget // 10)
        
        # Generate lattice points in [-4, 4]^dim (minima are in [-4,4]^5 per contract)
        # Use a subset of the integer lattice
        lattice_points = []
        # Add center
        lattice_points.append(np.zeros(dim))
        
        # Add points on a coarser grid first, then finer
        for step in [2.0, 1.0, 0.5]:
            if len(lattice_points) >= n_lattice:
                break
            # Generate grid points
            grid_vals = np.arange(-4.0, 4.5, step)
            # Use itertools.product equivalent via meshgrid for dim=5
            if dim <= 5:
                grids = np.meshgrid(*([grid_vals] * dim), indexing='ij')
                for i in range(len(grid_vals) ** dim):
                    if len(lattice_points) >= n_lattice:
                        break
                    # Flatten index to multi-index
                    idx = np.unravel_index(i, (len(grid_vals),) * dim)
                    point = np.array([grids[d][idx[d]] for d in range(dim)])
                    # Add small random jitter to avoid exact duplicates and explore around lattice
                    point += rng.uniform(-0.1, 0.1, dim)
                    lattice_points.append(point)
        
        # Evaluate lattice points
        for x0 in lattice_points:
            fx0 = safe_call(x0)
            if fx0 is None:
                break

        # Phase 2: CMA-ES style population-based refinement
        # Use remaining budget for a population-based search
        if x_best is not None and calls < budget:
            x = x_best.copy()
            fx = fx_best
            
            # Population size
            pop_size = min(12, max(4, (budget - calls) // 20))
            sigma = 1.0  # Initial step size
            
            # CMA-ES simplified: maintain a population
            # For simplicity and speed, use a (mu+lambda) ES with adaptation
            while calls < budget and time.time() < deadline:
                # Generate population
                offspring = []
                for _ in range(pop_size):
                    y = x + sigma * rng.standard_normal(dim)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    offspring.append((y, fy))
                
                if not offspring:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[1])
                
                # Select best
                best_y, best_fy = offspring[0]
                
                if best_fy < fx:
                    # Update mean
                    x = best_y
                    fx = best_fy
                    # Increase sigma slightly
                    sigma = min(sigma * 1.02, 5.0)
                else:
                    # No improvement, decrease sigma
                    sigma = max(sigma * 0.9, 0.01)
                    if sigma < 0.02:
                        break
                
                # Early stopping check
                if time.time() > deadline:
                    break

        return x_best, fx_best
