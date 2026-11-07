"""Multi-objective optimizer on WFG9 - targeted search with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Targeted multi-objective search with adaptive local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (hi + lo) / 2.0
        
        rng = np.random.default_rng(12345)
        
        # Phase 1: Informed initialization - sample near center and corners
        # WFG9 often has good solutions near the center of the search space
        n_init = min(30, self.budget)
        
        # Mix of center-biased and random initialization
        X_init = np.empty((n_init, self.dim))
        # 50% near center with small perturbations
        n_center = n_init // 2
        X_init[:n_center] = mid + span * rng.standard_normal((n_center, self.dim)) * 0.1
        # 30% moderate spread
        n_mid = n_init - n_center - (n_init // 10)
        X_init[n_center:n_center + n_mid] = lo + span * rng.random((n_mid, self.dim))
        # 20% full random
        n_rand = n_init - n_center - n_mid
        if n_rand > 0:
            X_init[n_center + n_mid:] = lo + span * rng.random((n_rand, self.dim))
        
        X_init = np.clip(X_init, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Track all evaluated points for hypervolume
        all_X = X_init.copy()
        all_F = F_init.copy()
        
        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for fj in F_set:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False
        
        def dominated_indices(F_set):
            """Return set of indices that are dominated."""
            n = len(F_set)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                for j in range(n):
                    if i == j or dom[j]:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dom[i] = True
                        break
            return np.where(~dom)[0]
        
        # Get initial non-dominated set
        nondom = dominated_indices(all_F)
        
        # Phase 2: Adaptive local search from non-dominated points
        # Use a shrinking step size with momentum
        step_scale = 0.05
        momentum = np.zeros(self.dim)
        velocity = 0.01
        
        # Sort non-dominated points by sum of objectives (proxy for quality)
        sums = np.sum(all_F[nondom], axis=1)
        order = np.argsort(sums)
        nondom_sorted = nondom[order]
        
        # Focus search on top performers
        n_search = min(len(nondom_sorted), 15)
        search_indices = nondom_sorted[:n_search]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive step size: decrease over time
            progress = used / self.budget
            step = step_scale * (1.0 - 0.7 * progress) * span
            
            # Select a parent from non-dominated set (weighted toward better)
            # Use tournament-like selection
            n_tourn = min(3, len(search_indices))
            candidates = rng.choice(search_indices, size=n_tourn, replace=False)
            parent_idx = candidates[np.argmin(np.sum(all_F[candidates], axis=1))]
            
            # Generate child with Gaussian perturbation + momentum
            parent = all_X[parent_idx]
            direction = rng.standard_normal(self.dim)
            
            # Add momentum from previous direction
            momentum = 0.3 * momentum + 0.7 * direction
            child = parent + step * momentum + velocity * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child is non-dominated
            current_nondom_F = all_F[nondom]
            is_dom = False
            for fj in current_nondom_F:
                if np.all(fj <= fc) and np.any(fj < fc):
                    is_dom = True
                    break
            
            if not is_dom:
                # Child is non-dominated, add to archive
                all_X = np.vstack([all_X, child])
                all_F = np.vstack([all_F, fc])
                
                # Update non-dominated set
                nondom = dominated_indices(all_F)
                
                # Update search indices
                if len(nondom) > 0:
                    sums = np.sum(all_F[nondom], axis=1)
                    order = np.argsort(sums)
                    nondom_sorted = nondom[order]
                    n_search = min(len(nondom_sorted), 15)
                    search_indices = nondom_sorted[:n_search]
                
                # Slightly reduce velocity for stability
                velocity *= 0.95
            else:
                # Child is dominated, try a smaller step in a different direction
                # Sometimes accept if it's close to the front
                best_dist = np.min(np.sum(np.abs(all_F[nondom] - fc), axis=1))
                if best_dist < 0.01:
                    # Close enough, add anyway for diversity
                    all_X = np.vstack([all_X, child])
                    all_F = np.vstack([all_F, fc])
                    nondom = dominated_indices(all_F)
        
        return all_X
# EVOLVE-BLOCK-END
