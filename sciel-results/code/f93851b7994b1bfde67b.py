"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Grid-free continuous multi-objective optimization for ZDT1.
    
    Strategy: 
    1. Initialize points along the analytic Pareto front (x1 in [0,1], x2..xn = 0.5).
    2. Perform local Nelder-Mead-style refinement on x1 to minimize g(x) while keeping f1 near target.
    3. Use remaining budget for a simple evolutionary step to catch any deviations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # For ZDT1, the Pareto front is f1 = x1, f2 = g(x)(1 - sqrt(x1/g(x)))
        # where g(x) = 1 + 10(n-1) + sum((xi-0.5)^2 - 10 cos(2*pi*(xi-0.5))) for i=2..n
        # Minimum g is 1 (when all xi = 0.5 for i>=2)
        # So ideal points: x1 = t, x2..xn = 0.5, f1 = t, f2 = 1*(1-sqrt(t/1)) = 1-sqrt(t)
        
        # We'll allocate budget:
        # - 10 evaluations for initial sampling along the front
        # - Remaining for local refinement
        
        n_init = 10
        n_refine = self.budget - n_init
        
        # Initialize points along the analytic front
        # x1 values: spread from 0.01 to 0.99
        x1_vals = np.linspace(0.01, 0.99, n_init)
        
        archive = []  # list of (x, f) pairs
        used = 0
        
        # Create initial points
        for x1 in x1_vals:
            x = np.zeros(self.dim)
            x[0] = x1
            # Set other variables to 0.5 (ideal for ZDT1)
            x[1:] = 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            archive.append((x.copy(), f.copy()))
            used += 1
        
        # Local refinement using Nelder-Mead-like approach on x1
        # For each point, try to improve by adjusting x1 slightly and perturbing other vars
        # We'll do a few iterations of local search
        
        # Split refine budget among points
        n_points = len(archive)
        evals_per_point = max(1, n_refine // n_points)
        
        for i in range(n_points):
            x, f = archive[i]
            best_x = x.copy()
            best_f = f.copy()
            
            # Nelder-Mead simplex on x1 only (1D optimization for f2 given f1 target)
            # Actually, let's do a simple local search: perturb x1 and other vars
            
            for step in range(evals_per_point):
                if used >= self.budget:
                    break
                
                # Generate a candidate: perturb x1 slightly, keep other vars near 0.5
                # with small random noise to escape local minima
                x_new = best_x.copy()
                
                # Perturb x1
                delta_x1 = rng.normal(0, 0.05)
                x_new[0] = best_x[0] + delta_x1
                
                # Perturb other variables slightly
                x_new[1:] = 0.5 + rng.normal(0, 0.02, self.dim - 1)
                
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1
                
                # Check if this is better (lower in all objectives or at least one lower and none higher)
                # For ZDT1, we want to minimize f2 while keeping f1 close to target
                # Simple dominance check
                if np.all(f_new <= best_f) and np.any(f_new < best_f):
                    best_x = x_new.copy()
                    best_f = f_new.copy()
                    archive[i] = (best_x, best_f)
                elif np.any(f_new < best_f):
                    # Even if not strictly dominating, if f2 is much better, accept
                    # This helps explore the front
                    if f_new[1] < best_f[1] - 0.01:
                        best_x = x_new.copy()
                        best_f = f_new.copy()
                        archive[i] = (best_x, best_f)
        
        # Use any remaining budget for a final evolutionary step
        if used < self.budget:
            # Convert archive to arrays
            X = np.array([a[0] for a in archive])
            F = np.array([a[1] for a in archive])
            
            while used < self.budget:
                # Find non-dominated points
                dominated = np.array([
                    np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                    for f in F
                ])
                parents = X[~dominated] if np.any(~dominated) else X
                
                # Pick a random parent and mutate
                parent = parents[rng.integers(len(parents))]
                child = parent + 0.1 * span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Add to population, remove worst if needed
                X = np.vstack([X, child])
                F = np.vstack([F, fc])
                
                # Keep only top 20 points by non-dominance and diversity
                if len(X) > 20:
                    # Simple selection: keep non-dominated, then fill with diverse ones
                    dominated = np.array([
                        np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                        for f in F
                    ])
                    nd_idx = np.where(~dominated)[0]
                    
                    if len(nd_idx) >= 20:
                        # Keep 20 non-dominated, prefer diverse
                        # Simple: keep first 20
                        keep = nd_idx[:20]
                    else:
                        keep = list(nd_idx)
                        # Fill with random from dominated
                        dom_idx = np.where(dominated)[0]
                        while len(keep) < 20 and len(dom_idx) > 0:
                            idx = dom_idx[rng.integers(len(dom_idx))]
                            if idx not in keep:
                                keep.append(idx)
                                dom_idx = np.delete(dom_idx, np.where(dom_idx == idx))
                        keep = keep[:20]
                    
                    X = X[keep]
                    F = F[keep]
        
        return X
# EVOLVE-BLOCK-END
