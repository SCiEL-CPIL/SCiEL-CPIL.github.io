"""Rewritten program for DTLZ7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 using a structured search strategy.
    
    DTLZ7 has 3 objectives:
      f1 = x0
      f2 = x1
      f3 = (1 + g) * h(f1, f2, g)
    where g = 100 + 10*n + sum_{i in [2..n]} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
    and h = (m - f1 - f2 - sum_{i=1}^{m-2} (f_i * sin(pi * f_i / (2*(m-1)))))
    
    The Pareto front is defined by g=0 (all x_i >= 2 for i>=2) and f1, f2 in [0,1].
    On the front: f3 = h(f1, f2) where h is the DTLZ7 h function.
    
    Strategy:
    1. Initialize with points on/near the Pareto front (set decision variables for g to 0.5)
    2. Use a grid/quasi-random sampling on the Pareto front parameterization
    3. Refine with local search around promising regions
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim  # 22
        
        # DTLZ7 structure:
        # x[0] = f1, x[1] = f2, x[2:] = g variables
        # To be on Pareto front: g(x[2:]) = 0, which requires x[i] = 0.5 for i >= 2
        # (since (x-0.5)^2 - cos(20*pi*(x-0.5)) >= 0 with equality at x=0.5)
        
        # Phase 1: Sample points on the Pareto front using a structured grid
        # The Pareto front is parameterized by (f1, f2) in [0,1]^2
        # h(f1, f2) = m - f1 - f2 - sum_{i=1}^{m-2} (f_i * sin(pi * f_i / (2*(m-1))))
        # For 3 objectives (m=3): h = 3 - f1 - f2 - f1*sin(pi*f1/4) - f2*sin(pi*f2/4)
        # Wait, let me re-check. m = n_obj = 3.
        # h = (m - f1 - f2 - sum_{i=1}^{m-2} (f_i * sin(pi * f_i / (2*(m-1)))))
        # m-2 = 1, so sum is over i=1 only: f1 * sin(pi * f1 / (2*2)) = f1 * sin(pi*f1/4)
        # So h = 3 - f1 - f2 - f1*sin(pi*f1/4)
        
        # Actually for DTLZ7 with m objectives, the h function is:
        # h(u, g) = (m - sum_{i=1}^{m-1} u_i - sum_{i=1}^{m-2} u_i * sin(pi*u_i/(2*(m-1))))
        # where u_i are the first m-1 objectives.
        
        # For m=3: h = 3 - f1 - f2 - f1*sin(pi*f1/4)
        
        # Strategy: use a combination of:
        # 1. A coarse grid on the Pareto front
        # 2. Halton sequence for more uniform coverage
        # 3. Local refinement
        
        budget = self.budget
        points = []
        fvals = []
        used = 0
        
        def make_point_on_front(f1, f2):
            """Create a decision vector that lies on the Pareto front."""
            x = np.zeros(n)
            x[0] = f1
            x[1] = f2
            # Set remaining variables to 0.5 to make g=0
            x[2:] = 0.5
            # Clip to bounds
            x = np.clip(x, lo, hi)
            return x
        
        def evaluate_point(x):
            nonlocal used
            f = np.array(problem(x))
            used += 1
            return f
        
        # Phase 1: Halton sequence on the Pareto front for broad coverage
        # Use first ~60% of budget for exploration
        n_phase1 = int(budget * 0.6)
        
        def halton(index, base):
            """Generate Halton sequence value."""
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            return result
        
        # Generate Halton sequence points in 2D (for f1, f2)
        # Use primes 2 and 3 for the two dimensions
        for i in range(1, n_phase1 + 1):
            if used >= budget:
                break
            f1 = halton(i, 2)
            f2 = halton(i, 3)
            x = make_point_on_front(f1, f2)
            f = evaluate_point(x)
            points.append(x)
            fvals.append(f)
        
        # Phase 2: Local refinement around best regions
        # Find points with low f3 (since f1, f2 are already spread)
        # Use a simple local search: perturb points near the front
        
        remaining = budget - used
        if remaining > 0 and len(points) > 0:
            fvals_arr = np.array(fvals)
            
            # Sort by f3 (third objective) to find promising regions
            # But we also want diversity in f1, f2
            # Use non-dominated sorting to find good candidates
            
            def is_dominated(f, fvals_arr):
                """Check if f is dominated by any point in fvals_arr."""
                for g in fvals_arr:
                    if np.all(g <= f) and np.any(g < f):
                        return True
                return False
            
            # Find non-dominated points
            non_dom_idx = []
            for i, f in enumerate(fvals_arr):
                if not is_dominated(f, fvals_arr):
                    non_dom_idx.append(i)
            
            if not non_dom_idx:
                non_dom_idx = list(range(len(fvals_arr)))
            
            # Pick the most spread-out non-dominated points for local search
            # Use a simple approach: pick points that maximize spread
            n_local = min(remaining, 20)
            
            # Select diverse points from non-dominated set
            if len(non_dom_idx) > n_local:
                # Use k-means-like selection: pick points far apart
                selected = [non_dom_idx[0]]
                remaining_idx = non_dom_idx[1:]
                while len(selected) < n_local and remaining_idx:
                    best_idx = None
                    best_dist = -1
                    for idx in remaining_idx:
                        min_dist_to_selected = min(
                            np.linalg.norm(fvals_arr[idx] - fvals_arr[s]) for s in selected
                        )
                        if min_dist_to_selected > best_dist:
                            best_dist = min_dist_to_selected
                            best_idx = idx
                    selected.append(best_idx)
                    remaining_idx.remove(best_idx)
            else:
                selected = non_dom_idx
            
            # Local search around selected points
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            for idx in selected:
                if used >= budget:
                    break
                base_x = points[idx].copy()
                base_f = fvals_arr[idx].copy()
                
                # Try small perturbations in f1, f2 space
                for step in step_sizes:
                    if used >= budget:
                        break
                    # Perturb f1 and f2
                    for delta_f1 in [-step, 0, step]:
                        if used >= budget:
                            break
                        for delta_f2 in [-step, 0, step]:
                            if used >= budget:
                                break
                            if delta_f1 == 0 and delta_f2 == 0:
                                continue
                            new_f1 = base_x[0] + delta_f1
                            new_f2 = base_x[1] + delta_f2
                            if new_f1 < 0 or new_f1 > 1 or new_f2 < 0 or new_f2 > 1:
                                continue
                            x = make_point_on_front(new_f1, new_f2)
                            f = evaluate_point(x)
                            # Only keep if it's non-dominated
                            if not is_dominated(f, np.array(fvals)):
                                points.append(x)
                                fvals.append(f)
        
        return points
# EVOLVE-BLOCK-END
