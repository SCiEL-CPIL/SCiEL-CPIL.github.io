"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        
        # State tracking
        calls_made = 0
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # Helper to check budget and time
        def can_proceed():
            nonlocal calls_made
            if calls_made >= self.budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # 1. Initial exploration: Deterministic multi-start
        # We use a fixed set of starting points to ensure reproducibility and cover the box
        # Rosenbrock minimum is usually near (1, 1, ...), but hidden shifts/rotations apply.
        # We try a few strategic points: center, corners, and a point near the typical Rosenbrock minimum (1,1,1,1,1)
        
        start_points = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.full(self.dim, 0.5),
            np.full(self.dim, -0.5),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0])[:self.dim],
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0])[:self.dim],
            np.random.default_rng(42).uniform(-5, 5, self.dim),
            np.random.default_rng(123).uniform(-5, 5, self.dim),
        ]
        
        # Evaluate initial points to find a good starting basin
        # We reserve budget for optimization, so we limit initial evaluations
        max_initial_evals = min(8, self.budget // 4) 
        
        for p in start_points[:max_initial_evals]:
            if not can_proceed():
                break
            x = np.clip(p, -5, 5)
            f = func(x)
            calls_made += 1
            update_best(x, f)
            
        if not can_proceed():
            return best_x, best_f

        # 2. Nelder-Mead Optimization from the best starting point
        # We perform multiple restarts if budget allows, but prioritize the best initial point first.
        
        def nelder_mead(x0, f0, max_iter=50):
            nonlocal calls_made, best_x, best_f
            if not can_proceed():
                return
            
            # Initialize simplex
            x0 = np.clip(x0, -5, 5)
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Add other vertices
            for i in range(self.dim):
                xi = x0.copy()
                # Initial step size: 0.05 or 5% of range
                step = 0.05
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                fvals.append(fi)
                update_best(xi, fi)
                
                if not can_proceed():
                    return

            # Nelder-Mead Loop
            for _ in range(max_iter):
                if not can_proceed():
                    break
                
                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update global best
                if fvals[0] < best_f:
                    best_x = simplex[0].copy()
                    best_f = fvals[0]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                update_best(xr, fr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    update_best(xe, fe)
                    
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    update_best(xc, fc)
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            update_best(simplex[i], fi)
                            fvals[i] = fi
                            if not can_proceed():
                                return

        # Run Nelder-Mead from the best initial point
        nelder_mead(best_x, best_f)
        
        # 3. Coordinate Descent Polish
        # If we have budget left, polish with coordinate descent
        if can_proceed():
            x = best_x.copy()
            f = best_f
            step = 0.1
            # Simple coordinate descent with decreasing step
            for _ in range(20): # Max 20 iterations of CD
                if not can_proceed():
                    break
                improved = False
                for i in range(self.dim):
                    if not can_proceed():
                        break
                    # Try small steps in +/- direction
                    for direction in [1, -1]:
                        xn = x.copy()
                        xn[i] += direction * step
                        xn = np.clip(xn, -5, 5)
                        fn = func(xn)
                        calls_made += 1
                        if fn < f:
                            x = xn
                            f = fn
                            improved = True
                            update_best(x, f)
                            break # Move to next dimension if improved
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
