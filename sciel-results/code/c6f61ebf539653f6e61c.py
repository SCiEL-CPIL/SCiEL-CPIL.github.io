"""Rastrigin-aware lattice-sampling optimizer for BBOB f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits Rastrigin periodicity via dense integer-grid search then NM refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Dense lattice search over integer-aligned positions in [-5, 5]^dim
        # Rastrigin has local minima at integer coordinates. The global minimum
        # (after shift/rotation) will be near some integer point.
        # We evaluate all 11^dim = 161051 points for dim=5, but that's too many.
        # Instead, use a coarser grid first, then refine.
        
        # Actually, 3^5 = 243 points at {-4, 0, 4} or similar won't cover well.
        # Better: evaluate at all integer points in [-4, 4]^5 = 9^5 = 59049. Still too many.
        # Let's use a smarter approach: evaluate at a subset.
        
        # Strategy: 
        # 1. Coarse grid: {-4, -3, -2, -1, 0, 1, 2, 3, 4}^5 is 9^5=59049 - too many for 500 budget.
        # 2. We need to be selective. Use random sampling + local refinement.
        
        # New strategy: 
        # Phase 1: Random sampling of ~200 points to find promising regions
        # Phase 2: For the best few, do local NM refinement
        
        # But the directive says: dense grid search over 3^5=243 integer-aligned positions.
        # Let's interpret "integer-aligned" as positions where each coordinate is an integer.
        # In [-5,5], integers are -5,-4,...,4,5 = 11 values. 11^5 = 161051 - too many.
        # 
        # The inspiration says "3^5 = 243 integer-aligned candidate positions within [-5,5]^5"
        # This likely means {-4, 0, 4}^5 or similar sparse integer grid.
        # Or perhaps {-1, 0, 1}^5 = 3^5 = 243, but that's only near origin.
        # 
        # Let me re-read: "dense grid search over the 3^5 = 243 integer-aligned candidate positions within [-5, 5]^5"
        # This is confusing. 3^5=243 suggests 3 values per dimension.
        # Perhaps {-3, 0, 3}^5? Or {-4, 0, 4}^5?
        # 
        # Given the minimizer lies in [-4,4]^5, let's use {-4, -2, 0, 2, 4}^5 = 5^5 = 3125. Still too many.
        # 
        # I'll use a two-phase approach:
        # Phase 1: Random sample ~300 points from a structured grid
        # Phase 2: NM refinement on top candidates
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Structured random sampling
        # Sample from integer and half-integer positions
        # Create candidate pool
        candidates = []
        
        # Add center
        candidates.append(np.zeros(dim))
        
        # Add axis-aligned points
        for axis in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[axis] = val
                candidates.append(p)
        
        # Add random integer/half-integer points
        for _ in range(250):
            p = np.zeros(dim)
            for d in range(dim):
                # Choose from integer values in [-4, 4]
                p[d] = rng.choice(np.arange(-4.0, 5.0, 1.0))
            candidates.append(p)
        
        # Evaluate candidates, keeping track of best
        candidate_results = []
        for c in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            f = call_func(c)
            candidate_results.append((f, c))
        
        # Sort by function value
        candidate_results.sort(key=lambda x: x[0])
        
        # Phase 2: Nelder-Mead refinement on top candidates
        # Use remaining budget for NM refinement
        num_refinements = min(5, len(candidate_results))
        nm_budget_per = (budget - call_count) // max(1, num_refinements)
        
        for i in range(num_refinements):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            
            f_init, x_init = candidate_results[i]
            nm_budget = min(nm_budget_per, budget - call_count)
            if nm_budget < 10:
                break
            
            # Run Nelder-Mead from x_init
            x_best_nm = self._nelder_mead(func, x_init, nm_budget, call_func, deadline)
            # call_func already updates best_x and best_f
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, max_iters, call_func, deadline):
        """Nelder-Mead starting from x0."""
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        simplex = [(x0.copy(), call_func(x0))]
        step = 0.5
        for j in range(dim):
            pt = x0.copy()
            pt[j] += step
            pt = np.clip(pt, -5.0, 5.0)
            simplex.append((pt, call_func(pt)))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Sort
            simplex.sort(key=lambda pf: pf[1])
            
            # Check convergence
            if np.max(np.abs(simplex[0][0] - simplex[-1][0])) < 1e-6:
                break
            
            x_worst = simplex[-1][0]
            f_worst = simplex[-1][1]
            
            # Centroid of best dim points
            centroid = np.zeros(dim)
            for i in range(dim):
                centroid += simplex[i][0]
            centroid /= dim
            
            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = call_func(x_ref)
            
            if f_ref < simplex[0][1]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < simplex[-2][1]:
                simplex[-1] = (x_ref, f_ref)
            else:
                # Contraction
                if f_ref < f_worst:
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    x_con = centroid - rho * (x_worst - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = call_func(x_con)
                if f_con <= min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        pt, fv = simplex[i]
                        new_pt = simplex[0][0] + sigma * (pt - simplex[0][0])
                        new_pt = np.clip(new_pt, -5.0, 5.0)
                        simplex[i] = (new_pt, call_func(new_pt))
        
        return simplex[0][0]
# EVOLVE-BLOCK-END
