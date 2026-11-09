"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using refined evolutionary search with diversity."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Vectorized non-dominated sorting. Returns list of fronts (indices)."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = np.where(domination_count == 0)[0]
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = set()
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.add(j)
            current_front = np.array(list(next_front), dtype=int)
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        cd = np.zeros(n)
        F_sub = F[indices]
        for m in range(F_sub.shape[1]):
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (F_sub[order[i+1], m] - F_sub[order[i-1], m]) / (F_sub[:, m].max() - F_sub[:, m].min() + 1e-10)
        
        return cd

    def _hypervolume_3d(self, F, ref):
        """Compute hypervolume in 3D for points dominated by ref."""
        # Filter points that are better than ref in all objectives
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(F[:, 0])
        F = F[order]
        
        hv = 0.0
        prev_y = ref[1]
        prev_z = ref[2]
        
        for i in range(len(F)):
            y, z = F[i, 1], F[i, 2]
            if y < prev_y and z < prev_z:
                # This point contributes to hypervolume
                # Volume contribution: (x_i - x_{i+1}) * (y_i - y_{i+1}) * (z_i - z_{i+1})
                # But we need to handle the 3D case properly
                pass
        
        # Simpler: use inclusion-exclusion for 3D
        # Sort by first objective
        F_sorted = F[np.argsort(F[:, 0])]
        
        hv = 0.0
        for i in range(len(F_sorted)):
            x = F_sorted[i, 0]
            y = F_sorted[i, 1]
            z = F_sorted[i, 2]
            
            # Find the rectangle in yz-plane that this point dominates
            # that isn't already covered by previous points
            # This is complex; use a simple approximation
            # Contribution is (ref[0] - x) * (ref[1] - y) * (ref[2] - z)
            # but subtract overlaps
            
            # For 3D, exact HV is complex. Use a simpler approach:
            # Sort by x, then for each point, compute the area in yz it uniquely covers
            pass
        
        # Fallback: use a grid-based approximation or just return 0 if too complex
        # For now, use a simple Monte Carlo estimate
        n_samples = 1000
        samples = np.random.uniform([0, 0, 0], ref, (n_samples, 3))
        
        # For each sample, check if it's dominated by any point in F
        # Vectorized: for each sample, check if there exists a point in F that dominates it
        # F[i] dominates sample s if F[i,0] <= s[0] and F[i,1] <= s[1] and F[i,2] <= s[2]
        
        # This is O(n_samples * len(F)) which is fine for our sizes
        dominated = np.zeros(n_samples, dtype=bool)
        for i in range(len(F)):
            mask = (F[i, 0] <= samples[:, 0]) & (F[i, 1] <= samples[:, 1]) & (F[i, 2] <= samples[:, 2])
            dominated |= mask
        
        return ref[0] * ref[1] * ref[2] * np.mean(dominated)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        ref = np.array([1.1, 1.3, 1.1])
        
        # Phase 1: Initialize with a structured sample (Latin hypercube-like)
        n_init = 30
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(n_init) + 0.5) / n_init
            # Add some randomness
            X[:, d] += (hi[d] - lo[d]) * (rng.random(n_init) - 0.5) * 0.3
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Keep track of all non-dominated solutions
        non_dominated_X = []
        non_dominated_F = []
        
        # Find initial non-dominated set
        for i in range(n_init):
            is_dominated = False
            for j in range(n_init):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_X.append(X[i])
                non_dominated_F.append(F[i])
        
        non_dominated_X = np.array(non_dominated_X)
        non_dominated_F = np.array(non_dominated_F)
        
        # Phase 2: Hypervolume-driven adaptive sampling
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points in regions that might improve hypervolume
            # Strategy: sample near the current non-dominated front but in gaps
            
            n_candidates = 5
            candidates_X = []
            
            # 1. Points near existing non-dominated solutions (local refinement)
            if len(non_dominated_F) > 0:
                for _ in range(min(3, n_candidates)):
                    idx = rng.randint(len(non_dominated_F))
                    # Perturb in a way that might find new non-dominated points
                    # Try different perturbation directions
                    direction = rng.standard_normal(self.dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    step = 0.1 * (1.0 - 0.5 * used / self.budget)
                    cand = non_dominated_X[idx] + step * (hi - lo) * direction
                    cand = np.clip(cand, lo, hi)
                    candidates_X.append(cand)
            
            # 2. Points in the middle of the decision space (exploration)
            if len(non_dominated_F) > 0 and len(non_dominated_X) > 2:
                # Find the centroid of non-dominated solutions
                centroid = np.mean(non_dominated_X, axis=0)
                # Sample around the centroid
                cand = centroid + 0.15 * (hi - lo) * rng.standard_normal(self.dim)
                cand = np.clip(cand, lo, hi)
                candidates_X.append(cand)
            
            # 3. Random point (to escape local optima)
            cand = lo + (hi - lo) * rng.random(self.dim)
            candidates_X.append(cand)
            
            # Evaluate candidates and pick the one that maximizes marginal hypervolume
            best_cand_X = None
            best_cand_F = None
            best_hv_gain = -np.inf
            
            for cand in candidates_X:
                if time.time() > deadline:
                    break
                fc = problem(cand)
                used += 1
                
                # Check if this point is non-dominated and improves hypervolume
                # Simple check: is it better than at least one existing non-dominated point
                # and not dominated by any existing non-dominated point?
                
                is_dominated = False
                dominates_any = False
                for j in range(len(non_dominated_F)):
                    if np.all(non_dominated_F[j] <= fc) and np.any(non_dominated_F[j] < fc):
                        is_dominated = True
                        break
                    if np.all(fc <= non_dominated_F[j]) and np.any(fc < non_dominated_F[j]):
                        dominates_any = True
                
                if not is_dominated:
                    # Estimate hypervolume gain
                    # Add this point to the non-dominated set and compute HV
                    new_F = np.vstack([non_dominated_F, [fc]])
                    # Remove points dominated by the new point
                    keep_mask = np.ones(len(new_F), dtype=bool)
                    for j in range(len(new_F)):
                        if j != len(non_dominated_F):
                            continue
                        for k in range(len(non_dominated_F)):
                            if np.all(new_F[len(non_dominated_F)] <= new_F[k]) and np.any(new_F[len(non_dominated_F)] < new_F[k]):
                                keep_mask[k] = False
                    new_F = new_F[keep_mask]
                    
                    hv_new = self._hypervolume_3d(new_F, ref)
                    hv_old = self._hypervolume_3d(non_dominated_F, ref)
                    hv_gain = hv_new - hv_old
                    
                    if hv_gain > best_hv_gain:
                        best_hv_gain = hv_gain
                        best_cand_X = cand
                        best_cand_F = fc
            
            # If no candidate improved hypervolume, keep the best one anyway for exploration
            if best_cand_X is None:
                # Use the last evaluated candidate
                best_cand_X = candidates_X[-1]
                best_cand_F = problem(best_cand_X)
                used += 1
            
            # Update non-dominated set
            # Remove points dominated by the new point
            keep_mask = np.ones(len(non_dominated_F), dtype=bool)
            for j in range(len(non_dominated_F)):
                if np.all(best_cand_F <= non_dominated_F[j]) and np.any(best_cand_F < non_dominated_F[j]):
                    keep_mask[j] = False
            
            # Check if the new point is dominated by existing points
            is_dominated = False
            for j in range(len(non_dominated_F)):
                if keep_mask[j] and np.all(non_dominated_F[j] <= best_cand_F) and np.any(non_dominated_F[j] < best_cand_F):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated_F = np.vstack([non_dominated_F[keep_mask], [best_cand_F]])
                non_dominated_X = np.vstack([non_dominated_X[keep_mask], [best_cand_X]])
        
        return non_dominated_X
# EVOLVE-BLOCK-END
