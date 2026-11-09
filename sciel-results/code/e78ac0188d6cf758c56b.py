"""Multi-start Nelder-Mead optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start Nelder-Mead simplex optimizer.
    
    Uses 10 diverse initial centroids to seed simplices, then performs
    aggressive local refinement via the Nelder-Mead geometric method.
    This exploits the step-ellipsoidal structure by contracting simplices
    directly toward discontinuities without covariance adaptation overhead.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, keep 15s safety margin
        budget = self.budget
        dim = self.dim

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                f = float(f)
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 1: Generate 10 diverse starting points and evaluate them ---
        n_starts = 10
        start_points = []
        
        # Deterministic diverse points
        start_points.append(np.zeros(dim))
        start_points.append(np.ones(dim) * 2.0)
        start_points.append(-np.ones(dim) * 2.0)
        start_points.append(np.ones(dim) * 3.5)
        start_points.append(-np.ones(dim) * 3.5)
        
        # Random points
        for _ in range(n_starts - 5):
            start_points.append(rng.uniform(-4.5, 4.5, dim))
        
        # Evaluate all starting points
        start_fvals = []
        for sp in start_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = safe_call(sp)
            start_fvals.append(f)
        
        # Sort by fitness (best first)
        if start_fvals:
            indices = np.argsort(start_fvals)
            start_points = [start_points[i] for i in indices]
            start_fvals = [start_fvals[i] for i in indices]

        # --- Phase 2: Nelder-Mead from each starting point ---
        # Allocate remaining budget among starts, weighted toward better starts
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Distribute budget: top 3 starts get more evaluations
        weights = [3.0, 2.5, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        total_weight = sum(weights[:min(n_starts, len(start_points))])
        
        for s in range(min(n_starts, len(start_points))):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Budget for this start
            alloc = int(remaining_budget * weights[s] / total_weight)
            alloc = max(alloc, 20)  # Minimum 20 evaluations per start
            alloc = min(alloc, budget - calls_made)
            
            if alloc <= 0:
                break
            
            x0 = start_points[s].copy()
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = safe_call(x0)
            
            if not np.isfinite(f0):
                continue
            
            # Build initial simplex around x0
            simplex = [x0]
            simplex_fvals = [f0]
            
            # Add dim vertices offset from x0
            for d in range(dim):
                v = x0.copy()
                # Offset in dimension d
                offset = 0.5 if d % 2 == 0 else -0.5
                v[d] += offset
                v = np.clip(v, -5.0, 5.0)
                fv = safe_call(v)
                simplex.append(v)
                simplex_fvals.append(fv)
            
            # Run Nelder-Mead
            simplex, simplex_fvals = self._nelder_mead(
                simplex, simplex_fvals, safe_call, alloc, deadline, dim
            )
            
            # Update global best
            if simplex_fvals:
                local_best_idx = np.argmin(simplex_fvals)
                local_best_x = simplex[local_best_idx]
                local_best_f = simplex_fvals[local_best_idx]
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
            
            # Update remaining budget
            remaining_budget = budget - calls_made
            if remaining_budget <= 0:
                break

        # --- Phase 3: Final refinement on the best point found ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Do a few more Nelder-Mead iterations centered on the best point
            final_alloc = min(50, budget - calls_made)
            if final_alloc > 5:
                simplex = [best_x.copy()]
                simplex_fvals = [safe_call(best_x)]
                
                for d in range(dim):
                    v = best_x.copy()
                    v[d] += 0.2 if d % 2 == 0 else -0.2
                    v = np.clip(v, -5.0, 5.0)
                    fv = safe_call(v)
                    simplex.append(v)
                    simplex_fvals.append(fv)
                
                simplex, simplex_fvals = self._nelder_mead(
                    simplex, simplex_fvals, safe_call, final_alloc, deadline, dim
                )
                
                if simplex_fvals:
                    local_best_idx = np.argmin(simplex_fvals)
                    local_best_x = simplex[local_best_idx]
                    local_best_f = simplex_fvals[local_best_idx]
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

        # Ensure we've made at least some calls
        if calls_made == 0:
            x0 = np.zeros(dim)
            safe_call(x0)

        return best_x, best_f

    def _nelder_mead(self, simplex, simplex_fvals, safe_call, max_evals, deadline, dim):
        """
        Nelder-Mead simplex method.
        
        Args:
            simplex: list of (dim+1) points
            simplex_fvals: list of function values
            safe_call: function to evaluate
            max_evals: max evaluations allowed
            deadline: time deadline
            dim: dimensionality
            
        Returns:
            (simplex, simplex_fvals) after optimization
        """
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        n = len(simplex)  # dim + 1
        evals_used = 0
        
        max_iter = max_evals  # Each iteration uses ~1-4 evaluations
        
        for it in range(max_iter):
            if evals_used >= max_evals or time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(simplex_fvals)
            simplex = [simplex[i] for i in order]
            simplex_fvals = [simplex_fvals[i] for i in order]
            
            # Check convergence
            if abs(simplex_fvals[-1] - simplex_fvals[0]) < 1e-10:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_fvals[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            evals_used += 1
            
            if fr < simplex_fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                evals_used += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_fvals[-1] = fr
            elif fr < simplex_fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_fvals[-1] = fr
            else:
                # Contraction
                # Check if we should do outside or inside contraction
                if fr < worst_f:
                    # Outside contraction
                    xc = centroid + rho * (centroid - worst)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst)
                
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                evals_used += 1
                
                if fc < min(fr, worst_f):
                    simplex[-1] = xc
                    simplex_fvals[-1] = fc
                else:
                    # Shrink: move all points (except best) toward best
                    best = simplex[0]
                    for i in range(1, n):
                        simplex[i] = best + sigma * (simplex[i] - best)
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        simplex_fvals[i] = fi
                        evals_used += 1
                        if evals_used >= max_evals or time.time() > deadline:
                            break
                    if evals_used >= max_evals or time.time() > deadline:
                        break
        
        return simplex, simplex_fvals
# EVOLVE-BLOCK-END
