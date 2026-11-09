"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with initial random basin selection."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        # --- Phase 1: Initial random scan to find a good starting point ---
        # Spend ~20% of budget on random sampling to locate a promising basin
        n_scan = max(10, budget // 5)
        best_x = np.zeros(dim)
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Always evaluate origin and a few strategic points
        candidates = [np.zeros(dim)]
        candidates.append(np.array([2.0] * dim))
        candidates.append(np.array([-2.0] * dim))
        for _ in range(n_scan - len(candidates)):
            candidates.append(rng.uniform(-4, 4, dim))
        
        for c in candidates:
            if time.time() > deadline:
                break
            fc = func(np.clip(c, -5, 5))
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        # --- Phase 2: Nelder-Mead simplex local search ---
        # Initialize simplex around best_x
        x0 = best_x.copy()
        simplex = [x0.copy()]
        fvals = [best_f]
        
        # Create initial simplex vertices
        init_step = 0.5
        for i in range(dim):
            v = x0.copy()
            v[i] += init_step
            simplex.append(v)
            fvals.append(func(v))
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        calls_made = n_scan + dim + 1  # approximate count
        # Actually track precisely: n_scan calls + dim+1 simplex init calls
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence: if best and worst are very close, break
            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break
                
            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_made += 1
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
