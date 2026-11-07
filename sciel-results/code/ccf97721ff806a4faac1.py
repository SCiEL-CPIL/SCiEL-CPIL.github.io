"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead Simplex Optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = float('inf')
        
        # Number of restarts
        # We allocate budget among restarts
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        
        for r in range(num_restarts):
            # Initialize simplex
            # Start point: random within [-3, 3] to allow exploration
            start = rng.uniform(-3, 3, dim)
            
            # Simplex initialization
            # Vertices: start + delta * basis vectors
            # We use a relative delta based on the domain size
            delta = 1.0
            simplex = [start.copy()]
            for i in range(dim):
                v = start.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for v in simplex:
                v_clipped = np.clip(v, -5, 5)
                fv = func(v_clipped)
                fvals.append(fv)
                
                if fv < best_fx:
                    best_fx = fv
                    best_x = v_clipped.copy()
                    
                budget_per_restart -= 1
                if budget_per_restart <= 0:
                    break
                    
            if budget_per_restart <= 0:
                break

            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            while budget_per_restart > 0:
                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check for convergence (simple heuristic)
                if len(fvals) > 1:
                    if abs(fvals[0] - fvals[-1]) < 1e-8:
                        break
                    if np.max(np.abs(np.array(simplex) - np.array(simplex[0]))) < 1e-8:
                        break

                # Centroid of all but the worst point
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                budget_per_restart -= 1
                
                if f_r < best_fx:
                    best_fx = f_r
                    best_x = x_r.copy()
                
                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    budget_per_restart -= 1
                    
                    if f_e < best_fx:
                        best_fx = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Try contraction
                    # Outside contraction if f_r >= f_worst, inside otherwise
                    if f_r < f_worst:
                        # Inside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        budget_per_restart -= 1
                        
                        if f_c < best_fx:
                            best_fx = f_c
                            best_x = x_c.copy()
                        
                        if f_c < f_r:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrinkage
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                budget_per_restart -= 1
                                if fvals[i] < best_fx:
                                    best_fx = fvals[i]
                                    best_x = simplex[i].copy()
                                if budget_per_restart <= 0:
                                    break
                    else:
                        # Outside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        budget_per_restart -= 1
                        
                        if f_c < best_fx:
                            best_fx = f_c
                            best_x = x_c.copy()
                        
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrinkage
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                budget_per_restart -= 1
                                if fvals[i] < best_fx:
                                    best_fx = fvals[i]
                                    best_x = simplex[i].copy()
                                if budget_per_restart <= 0:
                                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
