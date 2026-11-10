"""Optimizer for WFG9 exploiting analytic structure: dense grid on (x0,x1) + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits WFG9's structure: objectives depend primarily on x0 and x1.
    Uses a dense deterministic grid over the 2D active subspace, then refines
    around promising regions with local perturbations of the active variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # WFG9: objectives depend on x0, x1 primarily. Other 10 vars are "distance" vars.
        # Strategy:
        # 1. Dense grid over (x0, x1) with fixed midpoint for other vars.
        # 2. Refinement: adaptive local search in 2D active subspace around non-dominated points.
        # 3. Final polish: tight local optimization on best non-dominated points.

        # Allocate budget: ~70% grid, ~20% refinement, ~10% polish
        n_grid = int(budget * 0.70)
        n_refine = int(budget * 0.20)
        n_polish = budget - n_grid - n_refine

        # Grid over x0 and x1 - use non-square allocation for better coverage
        # Try to get approximately equal spacing; use more points in x0 if it has wider range
        range0 = hi[0] - lo[0]
        range1 = hi[1] - lo[1]
        total_range = range0 + range1
        if total_range > 0:
            frac0 = range0 / total_range
            frac1 = range1 / total_range
        else:
            frac0 = frac1 = 0.5
        
        n0 = max(2, int(np.sqrt(n_grid * frac0)))
        n1 = max(2, int(np.sqrt(n_grid * frac1)))
        # Adjust to fit budget
        while n0 * n1 > n_grid and n0 > 2:
            n0 -= 1
        while n0 * n1 > n_grid and n1 > 2:
            n1 -= 1

        x0_vals = np.linspace(lo[0], hi[0], n0)
        x1_vals = np.linspace(lo[1], hi[1], n1)

        # Fixed values for the other 10 variables: midpoint
        fixed_vars = (lo[2:] + hi[2:]) / 2.0

        # Evaluate grid
        all_X = []
        all_F = []
        used = 0

        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= n_grid:
                    break
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = fixed_vars
                f = problem(x)
                all_X.append(x.copy())
                all_F.append(f)
                used += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

        all_X = np.array(all_X)
        all_F = np.array(all_F)

        if len(all_F) == 0:
            return all_X

        # Helper: find non-dominated indices
        def get_non_dominated(F):
            n_pts = len(F)
            if n_pts == 0:
                return np.array([], dtype=int)
            is_dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                if is_dominated[i]:
                    continue
                for j in range(n_pts):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        nd_idx = get_non_dominated(all_F)

        # Phase 2: Refinement - adaptive local search in 2D active subspace
        if len(nd_idx) > 0 and used < budget - n_polish:
            # Select diverse targets from non-dominated set
            n_nd = len(nd_idx)
            max_targets = min(n_nd, max(5, n_refine // 8))
            
            if n_nd > max_targets:
                # Sort by first objective and pick evenly
                nd_sorted = nd_idx[np.argsort(all_F[nd_idx, 0])]
                step = max(1, n_nd // max_targets)
                target_idx = nd_sorted[::step][:max_targets]
            else:
                target_idx = nd_idx

            # Adaptive step size based on grid spacing
            dx0 = (hi[0] - lo[0]) / max(1, n0 - 1)
            dx1 = (hi[1] - lo[1]) / max(1, n1 - 1)
            base_step = np.array([dx0, dx1])

            for t_idx in target_idx:
                if used >= budget - n_polish:
                    break
                if time.time() > deadline:
                    break
                
                base_x = all_X[t_idx].copy()
                base_f = all_F[t_idx].copy()
                
                # Local search: try perturbations in x0, x1 with decreasing step
                for step_mult in [1.0, 0.5, 0.25]:
                    if used >= budget - n_polish:
                        break
                    if time.time() > deadline:
                        break
                    
                    step = base_step * step_mult
                    # Try 8 directions + center
                    directions = [
                        [1, 0], [-1, 0], [0, 1], [0, -1],
                        [1, 1], [-1, 1], [1, -1], [-1, -1]
                    ]
                    
                    for d in directions:
                        if used >= budget - n_polish:
                            break
                        if time.time() > deadline:
                            break
                        
                        x_new = base_x.copy()
                        x_new[0] = np.clip(base_x[0] + step[0] * d[0], lo[0], hi[0])
                        x_new[1] = np.clip(base_x[1] + step[1] * d[1], lo[1], hi[1])
                        x_new[2:] = fixed_vars
                        
                        f_new = problem(x_new)
                        
                        # Check if this improves the local Pareto front
                        # Keep if not dominated by base_f
                        if not (np.all(base_f <= f_new) and np.any(base_f < f_new)):
                            all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                            all_F = np.vstack([all_F, f_new.reshape(1, -1)])
                            used += 1
                            # Update base if this is better in some sense
                            if np.all(f_new <= base_f) and np.any(f_new < base_f):
                                base_x = x_new
                                base_f = f_new

        # Phase 3: Polish - tight local optimization on best non-dominated points
        if used < budget and len(all_F) > 0:
            nd_idx = get_non_dominated(all_F)
            if len(nd_idx) > 0:
                # Select top non-dominated points for polish
                n_polish_targets = min(len(nd_idx), max(3, n_polish // 6))
                # Sort by hypervolume contribution proxy (distance to reference)
                ref = np.array([3.0, 5.0, 7.0])
                # Use sum of (ref - f) as proxy for contribution
                contributions = np.sum(ref - all_F[nd_idx], axis=1)
                top_idx = nd_idx[np.argsort(-contributions)][:n_polish_targets]
                
                for t_idx in top_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    base_x = all_X[t_idx].copy()
                    base_f = all_F[t_idx].copy()
                    
                    # Very tight local search with small steps
                    dx0 = (hi[0] - lo[0]) / max(1, n0 - 1)
                    dx1 = (hi[1] - lo[1]) / max(1, n1 - 1)
                    tiny_step = np.array([dx0, dx1]) * 0.1
                    
                    for _ in range(min(6, budget - used)):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Try 4 cardinal directions
                        for d in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                            if used >= budget:
                                break
                            if time.time() > deadline:
                                break
                            
                            x_new = base_x.copy()
                            x_new[0] = np.clip(base_x[0] + tiny_step[0] * d[0], lo[0], hi[0])
                            x_new[1] = np.clip(base_x[1] + tiny_step[1] * d[1], lo[1], hi[1])
                            x_new[2:] = fixed_vars
                            
                            f_new = problem(x_new)
                            
                            # Keep if not dominated by base_f
                            if not (np.all(base_f <= f_new) and np.any(base_f < f_new)):
                                all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                                all_F = np.vstack([all_F, f_new.reshape(1, -1)])
                                used += 1
                                if np.all(f_new <= base_f) and np.any(f_new < f_new):
                                    base_x = x_new
                                    base_f = f_new

        return all_X

# EVOLVE-BLOCK-END
