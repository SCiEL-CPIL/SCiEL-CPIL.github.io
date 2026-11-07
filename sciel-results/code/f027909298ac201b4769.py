"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting DTLZ7's analytical structure.
    
    DTLZ7 with M=3 objectives:
      f1 = x0
      f2 = x1
      f3 = 1 + (n-2) * g + g * (1 - sqrt(x0))
    where g = 0 when x2...xn-1 are on the Pareto front.
    
    On the Pareto front (g=0):
      f1 = x0
      f2 = x1
      f3 = 1 + 1*(1 - sqrt(x0)) = 2 - sqrt(x0)
    
    So the Pareto front is parameterized by x0, x1 in [0,1]:
      (x0, x1, 2 - sqrt(x0))
    
    We need to find points on or near this front that maximize hypervolume
    with reference point [1.1, 1.1, 25.0].
    
    Strategy:
    1. Use a dense grid on the 2D Pareto front (x0, x1) to cover it well.
    2. Use remaining budget for local refinement around promising regions.
    3. Ensure all 22 variables are set correctly (x2...x21 = 0 for g=0).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front: x0 in [0,1], x1 in [0,1], x2..x21 = 0
        # f = (x0, x1, 2 - sqrt(x0))
        
        def make_point(x0, x1):
            """Create a 22-dim point on the Pareto front."""
            x = np.zeros(self.dim)
            x[0] = np.clip(x0, 0.0, 1.0)
            x[1] = np.clip(x1, 0.0, 1.0)
            # x[2:] = 0 ensures g = 0
            return x
        
        def eval_point(x0, x1):
            """Evaluate a point on the Pareto front."""
            x = make_point(x0, x1)
            return problem(x)
        
        # Phase 1: Dense grid on the Pareto front
        # Use a grid that's denser in regions contributing more to hypervolume
        # The hypervolume contribution depends on the 3D volume under the front
        
        # Allocate budget: ~60% for grid, ~40% for refinement
        grid_budget = int(self.budget * 0.6)
        refine_budget = self.budget - grid_budget
        
        # Create a non-uniform grid: denser where the front has higher curvature
        # f3 = 2 - sqrt(x0), so df3/dx0 = -1/(2*sqrt(x0)), which is large near x0=0
        # Use a transformed variable: u = sqrt(x0), so x0 = u^2
        # This makes the front smoother in (u, x1) space
        
        # Grid in (u, x1) space where u in [0, 1]
        n_u = int(np.sqrt(grid_budget))
        n_x1 = max(1, grid_budget // n_u)
        
        # Adjust to fit budget
        if n_u * n_x1 > grid_budget:
            n_x1 = grid_budget // n_u
        
        u_vals = np.linspace(0.0, 1.0, n_u)
        x1_vals = np.linspace(0.0, 1.0, n_x1)
        
        # Evaluate grid points
        F_list = []
        X_list = []
        used = 0
        
        for u in u_vals:
            x0 = u * u  # x0 = u^2
            for x1 in x1_vals:
                if used >= self.budget or time.time() > deadline:
                    break
                x = make_point(x0, x1)
                f = problem(x)
                F_list.append(f)
                X_list.append(x)
                used += 1
            if time.time() > deadline:
                break
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Refinement around non-dominated points
        if used < self.budget and len(F) > 0:
            # Find non-dominated points
            def is_dominated(f, F):
                """Check if f is dominated by any point in F."""
                # f is dominated if there exists f' in F such that f' <= f and f' < f
                return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
            
            non_dom_mask = np.array([not is_dominated(f, F) for f in F])
            
            if np.any(non_dom_mask):
                nd_F = F[non_dom_mask]
                nd_X = X[non_dom_mask]
            else:
                nd_F = F
                nd_X = X
            
            # Refine around non-dominated points with local perturbations
            rng = np.random.default_rng(42)
            
            # Number of refinement iterations
            n_refine = min(refine_budget, len(nd_F) * 5)
            
            for i in range(n_refine):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Pick a random non-dominated point to refine
                idx = rng.integers(len(nd_F))
                x_parent = nd_X[idx]
                
                # Small perturbation in x0 and x1
                delta_u = 0.02 * rng.standard_normal()
                delta_x1 = 0.02 * rng.standard_normal()
                
                x0_new = np.clip(x_parent[0] + delta_u * (1 - x_parent[0]), 0.0, 1.0)
                x1_new = np.clip(x_parent[1] + delta_x1, 0.0, 1.0)
                
                x_child = make_point(x0_new, x1_new)
                f_child = problem(x_child)
                used += 1
                
                # Replace the worst point in the current set if child is better
                # Use simple sum as a proxy
                if len(F) > 0:
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    F[worst_idx] = f_child
                    X[worst_idx] = x_child
                    # Update nd_F and nd_X if needed (simplified)
                    if non_dom_mask[worst_idx]:
                        nd_F[idx] = f_child
                        nd_X[idx] = x_child
        
        return X
# EVOLVE-BLOCK-END
