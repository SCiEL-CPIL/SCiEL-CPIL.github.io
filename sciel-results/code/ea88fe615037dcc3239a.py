# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol init, adaptive hybrid mutation, and polish phase."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Reference point for dominance checks (from task)
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Sobol sequence initial sampling for better uniformity ---
        from scipy.integrate import quad
        try:
            from scipy.stats import qmc
            init_n = 48  # power of 2 for Sobol
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            lhs = sobol.random(init_n)
        except ImportError:
            init_n = 50
            lhs = np.empty((init_n, dim))
            for d in range(dim):
                perm = rng.permutation(init_n)
                lhs[:, d] = (perm + rng.random(init_n)) / init_n

        X_init = lo + (hi - lo) * lhs
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = init_n

        # Keep all as archive
        archive_X = X_init.copy()
        archive_F = F_init.copy()

        # --- Helper functions ---
        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                dominated = np.any(
                    np.all(F[:i] <= F[i], axis=1) & np.any(F[:i] < F[i], axis=1)
                )
                if not dominated:
                    dominated = np.any(
                        np.all(F[i+1:] <= F[i], axis=1) & np.any(F[i+1:] < F[i], axis=1)
                    )
                if dominated:
                    mask[i] = False
            return mask

        def update_archive(archive_X, archive_F, child_x, child_f):
            """Check dominance and update archive. Returns (new_archive_X, new_archive_F, was_added)."""
            # Check if child is dominated by any in archive
            dominated_by_archive = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= child_f) and np.any(archive_F[j] < child_f):
                    dominated_by_archive = True
                    break

            if dominated_by_archive:
                return archive_X, archive_F, False

            # Check if child dominates any archive members
            new_archive_X = [child_x]
            new_archive_F = [child_f]
            for j in range(len(archive_F)):
                if not (np.all(child_f <= archive_F[j]) and np.any(child_f < archive_F[j])):
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])

            return np.array(new_archive_X), np.array(new_archive_F), True

        # Main search loop
        remaining = budget - used
        if remaining > 0:
            # Get initial non-dominated set
            nd_mask = non_dominated_mask(archive_F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(archive_F, axis=1))])

            # Sort by sum of objectives to get a starting point
            sum_obj = np.sum(archive_F, axis=1)
            best_idx = nd_idx[np.argmin(sum_obj[nd_idx])]

            # Phase 2: Adaptive hybrid mutation
            # Use most of the budget for this phase
            phase2_budget = int(remaining * 0.85)
            n_local = min(phase2_budget, remaining)

            for i in range(n_local):
                if time.time() > deadline - 15:
                    break
                if used >= budget:
                    break

                # Choose a center with hybrid strategy
                r = rng.random()
                if r < 0.35:
                    # Exploit: best point
                    center = archive_X[best_idx]
                elif r < 0.75:
                    # Explore: random non-dominated point
                    center = archive_X[nd_idx[rng.integers(len(nd_idx))]]
                else:
                    # Diversify: random from full archive
                    center = archive_X[rng.integers(len(archive_X))]

                # Adaptive step size with hybrid: mix of large and small steps
                progress = i / max(1, n_local - 1)
                base_step = 0.12 * (1.0 - 0.6 * progress) + 0.02

                # Hybrid: 70% small steps, 30% medium steps
                if rng.random() < 0.7:
                    step = base_step * 0.6
                else:
                    step = base_step * 1.5

                # Generate child
                child = center + step * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update archive
                archive_X, archive_F, was_added = update_archive(archive_X, archive_F, child, fc)

                if was_added:
                    # Update non-dominated indices
                    nd_mask = non_dominated_mask(archive_F)
                    nd_idx = np.where(nd_mask)[0]

                    # Update best index
                    if len(nd_idx) > 0:
                        sum_obj_new = np.sum(archive_F, axis=1)
                        best_idx = nd_idx[np.argmin(sum_obj_new[nd_idx])]

            # Phase 3: Polish phase - tighten around current non-dominated set
            polish_budget = budget - used
            if polish_budget > 0 and time.time() < deadline - 10:
                # Get current non-dominated set
                nd_mask = non_dominated_mask(archive_F)
                nd_idx = np.where(nd_mask)[0]
                if len(nd_idx) == 0:
                    nd_idx = np.array([np.argmin(np.sum(archive_F, axis=1))])

                for i in range(polish_budget):
                    if time.time() > deadline - 5:
                        break
                    if used >= budget:
                        break

                    # Pick a non-dominated point to polish around
                    center = archive_X[nd_idx[rng.integers(len(nd_idx))]]

                    # Very small steps for fine-tuning
                    step = 0.02 * rng.random() + 0.005

                    # Generate child
                    child = center + step * (hi - lo) * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)

                    fc = problem(child)
                    used += 1

                    # Update archive
                    archive_X, archive_F, was_added = update_archive(archive_X, archive_F, child, fc)

                    if was_added:
                        nd_mask = non_dominated_mask(archive_F)
                        nd_idx = np.where(nd_mask)[0]

        return archive_X

# EVOLVE-BLOCK-END
