"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _hv_contribution(self, F, ref):
        """Approximate hypervolume contribution using normalized distance to reference."""
        n = len(F)
        if n == 0:
            return np.array([])
        # Normalize objectives to [0,1] range based on current population
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        F_norm = (F - f_min) / f_range
        ref_norm = (ref - f_min) / f_range
        # Contribution: product of (ref - f) for each objective, normalized
        contribs = np.ones(n)
        for j in range(self.n_obj):
            contribs *= np.maximum(ref_norm[j] - F_norm[:, j], 0)
        # Add small epsilon to avoid zeros
        contribs = contribs + 1e-10
        return contribs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Initial population: mix of random and structured points
        pop_size = min(30, self.budget // 5)
        X = np.zeros((pop_size, self.dim))
        # First half: uniform random
        X[:pop_size//2] = lo + range_ * rng.random((pop_size//2, self.dim))
        # Second half: LHS-like structured (Latin hypercube approximation)
        if pop_size > pop_size//2:
            n_lhs = pop_size - pop_size//2
            for d in range(self.dim):
                perm = rng.permutation(n_lhs)
                X[pop_size//2:, d] = lo[d] + range_[d] * (perm + 0.5) / n_lhs
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (from task)
        ref = np.array([3.0, 5.0, 7.0])

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            non_dom_idx = self._non_dominated_indices(F)
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Select parent: weighted by HV contribution among non-dominated
            contribs = self._hv_contribution(F[non_dom_idx], ref)
            total = contribs.sum()
            if total <= 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                probs = contribs / total
                parent_idx = non_dom_idx[rng.choice(len(non_dom_idx), p=probs)]

            # Adaptive mutation
            progress = used / self.budget
            # Use a mix: mostly coordinate-wise, occasionally full vector
            if rng.random() < 0.3:
                # Coordinate-wise mutation
                n_mut = max(1, int(rng.integers(1, max(2, self.dim // 2))))
                mut_dims = rng.choice(self.dim, size=n_mut, replace=False)
                child = X[parent_idx].copy()
                for d in mut_dims:
                    step = 0.15 * (1 - 0.7 * progress) * range_[d]
                    child[d] += rng.standard_normal() * step
            else:
                # Full vector mutation with smaller scale
                scale = 0.12 * (1 - 0.7 * progress)
                child = X[parent_idx] + scale * range_ * rng.standard_normal(self.dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replacement: remove worst non-dominated or dominated point
            # Prefer removing dominated points; among non-dominated, remove least contributing
            all_indices = np.arange(len(F))
            dominated_mask = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_mask[i] = True
                        break

            if np.any(dominated_mask):
                # Remove a random dominated point
                dom_indices = np.where(dominated_mask)[0]
                worst = dom_indices[rng.integers(len(dom_indices))]
            else:
                # All non-dominated: remove least HV-contributing
                contribs_all = self._hv_contribution(F, ref)
                worst = int(np.argmin(contribs_all))

            X[worst] = child
            F[worst] = fc

        # Local polish: efficient coordinate-wise local search on top non-dominated points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            non_dom_idx = self._non_dominated_indices(F)
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))

            # Sort non-dominated by HV contribution, polish top ones
            contribs = self._hv_contribution(F[non_dom_idx], ref)
            order = np.argsort(-contribs)
            top_idx = non_dom_idx[order[:min(5, len(order))]]

            # Budget per polish point
            per_point = max(1, remaining // max(1, len(top_idx)))

            for idx in top_idx:
                if used >= self.budget or time.time() > deadline:
                    break

                best_x = X[idx].copy()
                best_f = F[idx].copy()

                # Multi-scale coordinate local search
                for step_frac in [0.08, 0.03, 0.01]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    step = step_frac * range_
                    improved = False
                    for dim_i in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            trial = best_x.copy()
                            trial[dim_i] += direction * step[dim_i]
                            trial = np.clip(trial, lo, hi)
                            if np.allclose(trial, best_x, atol=1e-12):
                                continue
                            trial_f = problem(trial)
                            used += 1
                            # Accept if it improves the worst objective or dominates
                            if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                                best_x = trial
                                best_f = trial_f
                                improved = True
                    if not improved:
                        break

                X[idx] = best_x
                F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
