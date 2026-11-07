"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Nelder-Mead + (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer combining Nelder-Mead simplex (geometric, derivative-free)
    with adaptive (1+1)-ES for final polishing.
    
    Strategy:
    1. Initialize with a diverse grid of 25 points.
    2. Run Nelder-Mead on each for a small fixed budget (20 evals) to identify promising basins.
    3. Take the best result and polish it with a robust (1+1)-ES using the 1/5 success rule.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Dense Grid Initialization + Nelder-Mead Screening
        # Generate 25 diverse starting points
        init_points = []
        
        # Center
        init_points.append(np.zeros(dim))
        
        # Axis-aligned points (positive and negative)
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 2.5
                init_points.append(p)
        
        # Diagonal-like points
        init_points.append(np.full(dim, 1.5))
        init_points.append(np.full(dim, -1.5))
        init_points.append(np.full(dim, 2.5))
        init_points.append(np.full(dim, -2.5))
        
        # Random diverse points to fill up to 25
        while len(init_points) < 25:
            init_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate each point and run a short Nelder-Mead
        # Budget per point: 20 evaluations (including the initial evaluation)
        nm_budget = 20
        
        for ip in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Initial evaluation
            fx_init = evaluate(ip)
            
            if fx_init < best_f:
                best_f = fx_init
                best_x = np.clip(ip, -5, 5).copy()
            
            # Run Nelder-Mead if we have budget
            if calls_made + nm_budget <= budget and time.time() < deadline:
                self._run_nelder_mead(func, ip.copy(), fx_init, nm_budget, deadline)
        
        # Phase 2: (1+1)-ES Polish
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate step size
            
            # 1/5 success rule parameters
            window_size = 20
            successes = []
            
            max_es_calls = budget - calls_made
            
            for i in range(max_es_calls):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                fy = evaluate(y)
                
                success = fy < fx
                successes.append(1 if success else 0)
                
                if len(successes) > window_size:
                    successes.pop(0)
                
                if success:
                    x = y
                    fx = fy
                else:
                    # Keep x, record failure
                    pass
                
                # Adapt sigma every window_size evaluations
                if len(successes) == window_size:
                    rate = sum(successes) / window_size
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 0.8
                    sigma = np.clip(sigma, 1e-5, 5.0)

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f

    def _run_nelder_mead(self, func, x_start, f_start, max_calls, deadline):
        """
        Run Nelder-Mead simplex method.
        """
        dim = self.dim
        x = x_start.copy()
        f = f_start
        
        # Initialize simplex
        # x0 is the current best
        simplex = [x.copy()]
        f_simplex = [f]
        
        # Add other vertices
        for i in range(dim):
            v = x.copy()
            # Step size 0.05 for local search, or 1.0 for more global?
            # Rastrigin has local minima at integers. A small step might get stuck.
            # Let's try 0.5 initially, scaled by dimension?
            # Standard NM uses 0.05 * |x_i| or 0.05 if x_i=0.
            # Given the scale [-5,5], 0.5 is reasonable for initial exploration.
            step = 0.5
            if abs(x[i]) < 1e-5:
                v[i] = step
            else:
                v[i] = x[i] + step
            simplex.append(v)
            fv = func(np.clip(v, -5, 5))
            self.calls_made += 1
            if fv < self.best_f:
                self.best_f = fv
                self.best_x = np.clip(v, -5, 5).copy()
            f_simplex.append(fv)
            
            if self.calls_made >= max_calls or time.time() > deadline:
                return

        # Main loop
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        while self.calls_made < max_calls and time.time() < deadline:
            # Sort simplex by fitness
            idx = np.argsort(f_simplex)
            simplex = [simplex[i] for i in idx]
            f_simplex = [f_simplex[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            self.calls_made += 1
            if f_r < self.best_f:
                self.best_f = f_r
                self.best_x = x_r.copy()
            
            if f_r < f_simplex[0]:
                # Expand
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                self.calls_made += 1
                if f_e < self.best_f:
                    self.best_f = f_e
                    self.best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contract
                # Outside contraction if f_r < f_worst
                if f_r < f_simplex[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (centroid - simplex[-1])
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                self.calls_made += 1
                if f_c < self.best_f:
                    self.best_f = f_c
                    self.best_x = x_c.copy()
                
                if f_c < f_simplex[-1]:
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        self.calls_made += 1
                        if f_simplex[i] < self.best_f:
                            self.best_f = f_simplex[i]
                            self.best_x = simplex[i].copy()
                        
                        if self.calls_made >= max_calls or time.time() > deadline:
                            return
# EVOLVE-BLOCK-END
