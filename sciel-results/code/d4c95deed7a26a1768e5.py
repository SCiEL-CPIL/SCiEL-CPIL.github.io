import numpy as np

class Optimizer:
    """
    Multi-Start Nelder-Mead Simplex Optimizer.
    Uses multiple random starting points and local simplex search to minimize
    the black-box function in 5 dimensions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, f0, budget_remaining, max_iter=50):
        """
        Run Nelder-Mead simplex optimization from x0 with f0.
        Returns (best_x, best_f, evals_used).
        """
        d = self.dim
        # Initialize simplex
        alpha = 0.5  # step size for initial simplex
        simplex = np.zeros((d + 1, d))
        fvals = np.zeros(d + 1)
        
        simplex[0] = x0
        fvals[0] = f0
        evals = 1
        
        for i in range(1, d + 1):
            x_i = x0.copy()
            x_i[i - 1] += alpha
            x_i = np.clip(x_i, -5, 5)
            simplex[i] = x_i
            fvals[i] = func(x_i)
            evals += 1
            if evals >= budget_remaining:
                break
        
        # Nelder-Mead parameters
        r = 1.0      # reflection
        e = 2.0      # expansion
        c = 0.5      # contraction
        s = 0.5      # shrink
        min_step = 1e-10
        
        for _ in range(max_iter):
            if evals >= budget_remaining:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check convergence
            f_range = fvals[-1] - fvals[0]
            x_range = np.max(np.abs(simplex - simplex[0]))
            if f_range < 1e-10 and x_range < min_step:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + r * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals += 1
            if evals >= budget_remaining:
                break
            
            if f_r < fvals[0]:
                # Try expansion
                x_e = centroid + e * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals += 1
                if evals >= budget_remaining:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + c * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - c * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                evals += 1
                if evals >= budget_remaining:
                    break
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        evals += 1
                        if evals >= budget_remaining:
                            break
                    if evals >= budget_remaining:
                        break
            
            # Track best
            best_idx = np.argmin(fvals)
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], evals

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        evals = 0
        
        # Phase 1: Coarse grid/random sampling to find promising regions
        # Use ~40% of budget for initial exploration
        n_initial = min(20, budget // 5)
        
        # Start with origin and a few random points
        candidates = [np.zeros(d)]
        
        # Add random points
        rng = np.random.RandomState(42)
        for _ in range(n_initial - 1):
            x = rng.uniform(-4, 4, size=d)
            candidates.append(x)
        
        # Evaluate initial candidates
        for x in candidates:
            if evals >= budget:
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped
        
        # Phase 2: Local refinement from best points using Nelder-Mead
        # Use remaining budget for local search
        # Try from the top few best initial points
        n_refine = min(5, budget - evals)
        budget_per_refine = max(1, (budget - evals) // max(1, n_refine))
        
        # Sort candidates by function value (we only have the best so far)
        # Try refining from best_x and a few other random points near it
        refine_starts = [best_x.copy()]
        
        # Add a few perturbations of the best point
        for _ in range(min(4, n_refine - 1)):
            x_pert = best_x + rng.normal(0, 0.5, size=d)
            x_pert = np.clip(x_pert, -5, 5)
            refine_starts.append(x_pert)
        
        for x_start in refine_starts:
            if evals >= budget:
                break
            
            # Evaluate starting point if not already done
            if np.allclose(x_start, best_x, atol=1e-6):
                f_start = best_f
            else:
                f_start = func(x_start)
                evals += 1
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start
            
            # Run Nelder-Mead
            x_opt, f_opt, evals_used = self._nelder_mead(
                func, x_start, f_start, budget - evals, max_iter=30
            )
            evals += evals_used
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
        
        # Phase 3: If we still have budget, do more random restarts
        while evals < budget - 5:
            # Random restart
            x_rand = rng.uniform(-4, 4, size=d)
            x_rand = np.clip(x_rand, -5, 5)
            f_rand = func(x_rand)
            evals += 1
            
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand
            
            # Quick local refinement
            if evals < budget - 10:
                x_opt, f_opt, evals_used = self._nelder_mead(
                    func, x_rand, f_rand, budget - evals, max_iter=15
                )
                evals += evals_used
                
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt
        
        # Final evaluation to ensure we're within budget
        if evals < budget:
            # Do a few more random evaluations
            for _ in range(budget - evals):
                x_rand = rng.uniform(-4, 4, size=d)
                x_rand = np.clip(x_rand, -5, 5)
                f_rand = func(x_rand)
                if f_rand < best_f:
                    best_f = f_rand
                    best_x = x_rand
                evals += 1
        
        return best_x, best_f
