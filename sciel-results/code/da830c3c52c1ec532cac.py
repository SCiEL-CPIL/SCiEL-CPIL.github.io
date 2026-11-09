"""Multi-objective optimizer on WFG4 with hypervolume-weighted island model."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Island model with hypervolume-weighted sub-regions for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random

        # ---- Define weight-based islands for 3-objective WFG4 ----
        # Each island focuses on a different region of the Pareto front
        n_islands = 4
        # Weight vectors: emphasize different objectives to cover the front
        weights = np.array([
            [0.8, 0.1, 0.1],  # emphasize obj1
            [0.1, 0.8, 0.1],  # emphasize obj2
            [0.1, 0.1, 0.8],  # emphasize obj3
            [0.34, 0.33, 0.33],  # balanced
        ])

        # Population per island
        pop_per_island = max(8, min(20, self.budget // (n_islands * 3)))
        total_init = pop_per_island * n_islands

        # ---- LHS initialization distributed across islands ----
        X = np.zeros((total_init, dim))
        F = np.zeros((total_init, n_obj))
        evals = 0

        for i in range(n_islands):
            start = i * pop_per_island
            end = start + pop_per_island
            # Generate LHS samples for this island
            lhs = np.zeros((pop_per_island, dim))
            for d in range(dim):
                bins = np.linspace(0, 1, pop_per_island + 1)
                pts = rng.uniform(bins[:-1], bins[1:])
                rng.shuffle(pts)
                lhs[:, d] = pts
            X[start:end] = lo + lhs * span
            for j in range(pop_per_island):
                F[start + j] = problem(X[start + j])
                evals += 1

        # Track best per objective
        best_obj = F.min(axis=0)

        # ---- Island evolution loop ----
        migration_interval = max(10, self.budget // 20)
        last_migrate = 0

        while evals < self.budget:
            if time.time() > deadline:
                break

            # Determine which island to evolve (round-robin)
            island_idx = (evals - last_migrate) % n_islands

            start = island_idx * pop_per_island
            end = start + pop_per_island
            pop_X = X[start:end].copy()
            pop_F = F[start:end].copy()

            # Weight for this island
            w = weights[island_idx]

            # ---- Weighted tournament selection ----
            # Compute scalarized fitness using island weights
            # Normalize objectives to [0,1] based on known reference
            ref = np.array([3.0, 5.0, 7.0])
            # Simple normalization: divide by reference (WFG4 objectives are positive)
            F_norm = pop_F / ref
            scalar = np.sum(F_norm * w, axis=1)

            # Selection: pick parent with best scalar fitness
            parent_idx = int(np.argmin(scalar))
            parent = pop_X[parent_idx].copy()

            # ---- Weight-specific mutation intensity ----
            # Islands emphasizing a single objective use smaller steps
            # to refine that region; balanced island uses larger steps
            max_weight = np.max(w)
            if max_weight > 0.5:
                # Focused island: smaller mutation
                sigma = 0.1 * span
            else:
                # Balanced island: larger mutation
                sigma = 0.3 * span

            # Gaussian mutation with weight-specific intensity
            child = parent + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            evals += 1

            # Update best per objective
            for k in range(n_obj):
                if fc[k] < best_obj[k]:
                    best_obj[k] = fc[k]

            # ---- Replacement: keep if it improves scalarized fitness
            # or adds diversity ----
            child_scalar = np.sum((fc / ref) * w)
            parent_scalar = scalar[parent_idx]

            # Replace if better or if it's non-dominated in this sub-population
            replace = False
            if child_scalar < parent_scalar - 1e-10:
                replace = True
            else:
                # Check if child is non-dominated by any in this island
                dominated = False
                for j in range(pop_per_island):
                    if j == parent_idx:
                        continue
                    if np.all(pop_F[j] <= fc) and np.any(pop_F[j] < fc):
                        dominated = True
                        break
                if not dominated:
                    # Find worst point in island to replace
                    worst_idx = int(np.argmax(scalar))
                    if worst_idx != parent_idx:
                        pop_X[worst_idx] = child
                        pop_F[worst_idx] = fc
                        X[start + worst_idx] = child
                        F[start + worst_idx] = fc
                        replace = True
                    else:
                        # Replace parent
                        pop_X[parent_idx] = child
                        pop_F[parent_idx] = fc
                        X[start + parent_idx] = child
                        F[start + parent_idx] = fc
                        replace = True

            if replace and not (child_scalar < parent_scalar - 1e-10):
                # Already handled above, skip
                pass
            elif child_scalar < parent_scalar - 1e-10:
                pop_X[parent_idx] = child
                pop_F[parent_idx] = fc
                X[start + parent_idx] = child
                F[start + parent_idx] = fc

            # ---- Periodic migration ----
            if evals - last_migrate >= migration_interval:
                last_migrate = evals
                # Migrate best individuals between adjacent islands
                for i in range(n_islands):
                    i_start = i * pop_per_island
                    i_end = i_start + pop_per_island
                    j = (i + 1) % n_islands
                    j_start = j * pop_per_island
                    j_end = j_start + pop_per_island

                    # Find best in island i by scalar fitness
                    Fi_norm = F[i_start:i_end] / ref
                    scalar_i = np.sum(Fi_norm * weights[i], axis=1)
                    best_in_i = i_start + int(np.argmin(scalar_i))

                    # Find worst in island j
                    Fj_norm = F[j_start:j_end] / ref
                    scalar_j = np.sum(Fj_norm * weights[j], axis=1)
                    worst_in_j = j_start + int(np.argmax(scalar_j))

                    # Migrate if better
                    if scalar_i[int(np.argmin(scalar_i))] < scalar_j[int(np.argmax(scalar_j))]:
                        X[worst_in_j] = X[best_in_i].copy()
                        F[worst_in_j] = F[best_in_i].copy()

        return X
# EVOLVE-BLOCK-END
