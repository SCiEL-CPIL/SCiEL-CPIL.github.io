"""Multi-objective optimizer on WFG7: Structure-aware search with diverse exploitation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search exploiting WFG7 structure with diverse local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        budget = self.budget
        dim = self.dim

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f, dtype=np.float64))
            return f

        # Phase 1: Structured initialization
        # WFG7 often has a flat Pareto front or specific trade-offs.
        # Start with a good spread.
        
        n_init = 40
        if budget >= n_init:
            X_init = lo + span * rng.random((n_init, dim))
            for x in X_init:
                if used >= budget or time.time() > deadline:
                    break
                evaluate(x)

        # Add structured points: vary one dimension at a time from the center
        # This helps find the "ridges" of the WFG landscape.
        n_struct_total = 60
        n_struct_per_dim = max(1, n_struct_total // dim)
        for d in range(dim):
            for _ in range(n_struct_per_dim):
                if used >= budget or time.time() > deadline:
                    break
                x = mid.copy()
                # Vary dimension d across its range
                x[d] = lo[d] + rng.random() * span[d]
                # Add small noise to other dimensions to avoid exact degeneracy
                x += 0.01 * span * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                evaluate(x)

        # Phase 2: Diverse Local Exploitation
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)

            if len(non_dom_idx) == 0:
                # Fallback: use points with lowest sum of objectives
                sums = np.sum(F_arr, axis=1)
                non_dom_idx = np.argsort(sums)[:min(10, len(F_arr))]

            # Select diverse points from the non-dominated set
            # Use a greedy maximin approach to select centers
            selected_centers = []
            remaining_indices = list(non_dom_idx)
            
            # Start with the point with the lowest sum (best compromise)
            if remaining_indices:
                best_start = min(remaining_indices, key=lambda i: np.sum(F_arr[i]))
                selected_centers.append(best_start)
                remaining_indices.remove(best_start)

            # Add more centers that are far from existing ones
            max_centers = 10
            while len(selected_centers) < max_centers and remaining_indices:
                if used >= budget or time.time() > deadline:
                    break
                best_idx = -1
                best_min_dist = -1
                for idx in remaining_indices:
                    f_point = F_arr[idx]
                    min_dist_to_selected = np.min([np.linalg.norm(f_point - F_arr[sc]) for sc in selected_centers])
                    if min_dist_to_selected > best_min_dist:
                        best_min_dist = min_dist_to_selected
                        best_idx = idx
                if best_idx == -1:
                    break
                selected_centers.append(best_idx)
                remaining_indices.remove(best_idx)

            # Exploit around these diverse centers
            # Allocate budget proportionally or evenly
            n_exploit_centers = len(selected_centers)
            if n_exploit_centers > 0:
                steps_per_center = max(1, (budget - used) // (n_exploit_centers * 2))
                # Cap steps to avoid spending too much on one center if others are left
                steps_per_center = min(steps_per_center, 15)

                for idx in selected_centers:
                    if used >= budget or time.time() > deadline:
                        break

                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    
                    # Adaptive local search
                    step_size = 0.05
                    for _ in range(steps_per_center):
                        if used >= budget or time.time() > deadline:
                            break

                        # Try 3 random directions per step
                        improved = False
                        for _ in range(3):
                            if used >= budget or time.time() > deadline:
                                break
                            perturbation = step_size * span * rng.standard_normal(dim)
                            trial_x = np.clip(center_x + perturbation, lo, hi)
                            trial_f = problem(trial_x)
                            used += 1
                            all_X.append(trial_x.copy())
                            all_F.append(np.array(trial_f, dtype=np.float64))

                            # Check for dominance improvement
                            if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                                center_x = trial_x
                                center_f = trial_f
                                improved = True

                        if improved:
                            step_size *= 0.8  # Shrink slightly on improvement
                        else:
                            step_size *= 0.5  # Shrink more on stagnation

        # Phase 3: Final Polish
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) > 0:
                # Pick the best few points (lowest sum) for final polish
                non_dom_idx = non_dom_idx[np.argsort(np.sum(F_arr[non_dom_idx], axis=1))]
                n_polish = min(5, len(non_dom_idx))
                for idx in non_dom_idx[:n_polish]:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.01
                    # Use remaining budget for a few fine steps
                    n_steps = min(10, (budget - used) // 2)
                    for _ in range(n_steps):
                        if used >= budget or time.time() > deadline:
                            break
                        perturbation = step_size * span * rng.standard_normal(dim)
                        trial_x = np.clip(center_x + perturbation, lo, hi)
                        trial_f = problem(trial_x)
                        used += 1
                        all_X.append(trial_x.copy())
                        all_F.append(np.array(trial_f, dtype=np.float64))
                        if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                            center_x = trial_x
                            center_f = trial_f
                        step_size *= 0.7

        # Return all evaluated points
        if len(all_F) > 0:
            return np.array(all_X)
        return np.array([])

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points using vectorized comparison."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        is_dominated = np.zeros(n, dtype=bool)

        # Vectorized dominance check
        # F[i] is dominated if exists j: F[j] <= F[i] and F[j] != F[i]
        for i in range(n):
            # F[j] <= F[i] for all objectives
            leq = np.all(F <= F[i], axis=1)
            # F[j] != F[i] (at least one strict)
            strict = np.any(F < F[i], axis=1)
            # Exclude self
            leq[i] = False
            if np.any(leq & strict):
                is_dominated[i] = True

        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
