"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-assisted Bayesian Optimization for BBOB-style Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 3:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        # Phase 1: Initial sampling with structured exploration
        # Use a mix of random and grid points to get good initial coverage
        rng = np.random.default_rng(12345)
        
        # Start with center and some structured points
        initial_points = []
        center = np.zeros(dim)
        initial_points.append(center)
        
        # Add points at various scales
        for scale in [4.0, 3.0, 2.0, 1.0, 0.5]:
            for d in range(dim):
                p = center.copy()
                p[d] = scale
                initial_points.append(p)
                p2 = center.copy()
                p2[d] = -scale
                initial_points.append(p2)
        
        # Add some random points
        for _ in range(20):
            initial_points.append(rng.uniform(-4.5, 4.5, size=dim))
        
        # Evaluate initial points
        for x in initial_points:
            if calls_made >= budget or time.time() > deadline - 5:
                break
            try:
                counted_func(x)
            except RuntimeError:
                break

        if best_x is None:
            # Fallback: evaluate center
            try:
                counted_func(np.zeros(dim))
            except RuntimeError:
                pass
            return best_x, best_f

        # Phase 2: Fit a surrogate model to estimate the transformation
        # The Katsuura function has a known structure: f(x) = 10/n * sum(1 - sqrt(cos(2*pi*x_i)))^2 + ...
        # With hidden shift s, rotation R, and offset o: f(x) = g(R*(x-s) + o) + c
        # We'll use the observed data to estimate s (shift) primarily
        
        # Collect all evaluated points and values
        # Since we don't have a history list, let's do a more direct approach:
        # Use the best point as an anchor and do local refinement
        
        # Phase 2: Multi-start local search using CMA-ES-like adaptation
        # Given the tight budget, we'll do a focused local search from the best point
        
        # Estimate a good step size based on the function's periodicity
        # Katsuura has period 2 in each dimension, so features are on scale ~1
        
        # Do a coordinate descent + local refinement
        # First, try a simple local search with adaptive step sizes
        
        # Local search phase
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        
        for step in step_sizes:
            if calls_made >= budget or time.time() > deadline - 5:
                break
            
            improved = True
            while improved and calls_made < budget and time.time() < deadline - 5:
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 5:
                        break
                    
                    # Try positive and negative steps in dimension d
                    for direction in [1, -1]:
                        x_try = best_x.copy()
                        x_try[d] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        
                        try:
                            f_try = counted_func(x_try)
                        except RuntimeError:
                            break
                        
                        if f_try < best_f:
                            improved = True
        
        # Phase 3: Final polish with Nelder-Mead
        if best_x is not None and calls_made < budget and time.time() < deadline - 5:
            n = dim
            simplex = [np.array(best_x, dtype=float, copy=True)]
            # Use a small step size for the final polish
            step_size = 0.05
            for i in range(n):
                xi = np.array(best_x, dtype=float, copy=True)
                xi[i] += step_size
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    break

            if len(fvals) == n + 1:
                alpha, gamma, rho = 1.0, 2.0, -0.5
                max_iterations = 100
                for _ in range(max_iterations):
                    if calls_made >= budget or time.time() > deadline - 3:
                        break
                    
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    
                    if max(fvals) - min(fvals) < 1e-12:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    try:
                        fr = counted_func(xr)
                    except RuntimeError:
                        break

                    if fr < fvals[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        try:
                            fe = counted_func(xe)
                        except RuntimeError:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        if fr < fvals[-1]:
                            xc = centroid + 0.5 * (xr - centroid)
                        else:
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        try:
                            fc = counted_func(xc)
                        except RuntimeError:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                try:
                                    fvals[i] = counted_func(simplex[i])
                                except RuntimeError:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
