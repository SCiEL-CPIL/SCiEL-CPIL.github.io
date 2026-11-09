"""Nelder-Mead based black-box optimizer for the rotated Sphere (BBOB f1)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead Simplex Method for 5-D continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        # Initial simplex
        # Center at origin
        center = np.zeros(d)
        # Initial step size
        # Since box is [-5, 5], a step of 1.0 or 2.0 is reasonable.
        # Let's use 1.0 to stay well within bounds initially.
        step = 1.0
        
        # Create simplex
        simplex = np.zeros((d + 1, d))
        # First vertex is center
        simplex[0] = center
        # Other vertices are center + step * e_i
        for i in range(1, d + 1):
            simplex[i] = center
            simplex[i, i-1] += step
        
        # Evaluate initial simplex
        fitness = np.zeros(d + 1)
        calls = 0
        for i in range(d + 1):
            if calls >= budget:
                break
            fitness[i] = func(simplex[i])
            calls += 1
        
        if calls >= budget:
            # Find best
            best_idx = np.argmin(fitness[:calls])
            return simplex[best_idx], fitness[best_idx]

        best_fit = np.min(fitness)
        best_pos = simplex[np.argmin(fitness)].copy()

        # Main loop
        while calls < budget:
            # Sort simplex by fitness
            order = np.argsort(fitness)
            simplex = simplex[order]
            fitness = fitness[order]
            
            # Check convergence (optional, but good for efficiency)
            # If range is small, we might be converged.
            if fitness[-1] - fitness[0] < 1e-10:
                break

            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            
            if calls >= budget:
                break
            f_r = func(x_r)
            calls += 1
            
            if f_r < fitness[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                
                if calls >= budget:
                    # Accept reflection if we can't evaluate expansion
                    simplex[-1] = x_r
                    fitness[-1] = f_r
                    if f_r < best_fit:
                        best_fit = f_r
                        best_pos = x_r.copy()
                    break
                
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fitness[-1] = f_e
                    if f_e < best_fit:
                        best_fit = f_e
                        best_pos = x_e.copy()
                else:
                    simplex[-1] = x_r
                    fitness[-1] = f_r
                    if f_r < best_fit:
                        best_fit = f_r
                        best_pos = x_r.copy()
            
            elif f_r < fitness[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fitness[-1] = f_r
                if f_r < best_fit:
                    best_fit = f_r
                    best_pos = x_r.copy()
            
            else:
                # Contraction
                # Outside contraction if f_r < fitness[-1], inside otherwise
                if f_r < fitness[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                else:
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                
                if calls >= budget:
                    break
                
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, fitness[-1]):
                    simplex[-1] = x_c
                    fitness[-1] = f_c
                    if f_c < best_fit:
                        best_fit = f_c
                        best_pos = x_c.copy()
                else:
                    # Shrinkage
                    # Shrink all points except the best towards the best
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        
                        if calls >= budget:
                            break
                        
                        fitness[i] = func(simplex[i])
                        calls += 1
                        if fitness[i] < best_fit:
                            best_fit = fitness[i]
                            best_pos = simplex[i].copy()
                    
                    if calls >= budget:
                        break
                    # Re-sort after shrinkage
                    order = np.argsort(fitness)
                    simplex = simplex[order]
                    fitness = fitness[order]
                    
                    # If we shrunk, we might need to continue
                    # But if we hit budget, we break

        return best_pos, best_fit
# EVOLVE-BLOCK-END
