# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Tchebycheff decomposition + simulated annealing refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # Reference point for scoring (not used in search, just for context)
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Generate Tchebycheff weight vectors ---
        # Create a diverse set of weight vectors on the simplex
        n_weights = 50
        # Generate random points on the simplex using Dirichlet
        weights = rng.dirichlet(np.ones(n_obj), size=n_weights)
        # Add some extreme points for better coverage
        extreme_weights = np.zeros((n_obj, n_obj))
        for i in range(n_obj):
            extreme_weights[i, i] = 1.0
        # Also add some intermediate weights
        mid_weights = np.zeros((n_obj * (n_obj - 1), n_obj))
        idx = 0
        for i in range(n_obj):
            for j in range(n_obj):
                if i != j:
                    mid_weights[idx, i] = 0.5
                    mid_weights[idx, j] = 0.5
                    idx += 1

        all_weights = np.vstack([weights, extreme_weights, mid_weights])
        # Normalize to ensure they sum to 1
        all_weights = all_weights / all_weights.sum(axis=1, keepdims=True)

        # We'll use a subset of these weights, prioritizing diversity
        # For 200 budget: use ~100 for initialization, ~100 for refinement
        n_init = 100
        n_refine = budget - n_init

        if n_init > len(all_weights):
            n_init = len(all_weights)

        # Select n_init weights with good spread
        # Use a greedy selection to maximize minimum distance
        selected_indices = [0]
        for _ in range(1, n_init):
            best_idx = -1
            best_min_dist = -1
            for i in range(len(all_weights)):
                if i in selected_indices:
                    continue
                min_dist = min(np.linalg.norm(all_weights[i] - all_weights[j])
                              for j in selected_indices)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            if best_idx >= 0:
                selected_indices.append(best_idx)

        init_weights = all_weights[selected_indices]

        # --- Phase 1: Evaluate each weight vector with a few local searches ---
        archive_X = []
        archive_F = []
        used = 0

        def dominates(f1, f2):
            """Check if f1 dominates f2 (all objectives <= and at least one <)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def update_archive(archive_X, archive_F, x, f):
            """Update archive with new point if non-dominated. Returns updated archives."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F, False

            # Remove dominated points from archive
            new_X = [x]
            new_F = [f]
            for i in range(len(archive_F)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])

            return new_X, new_F, True

        # For each weight, do a short local search to find a good point
        evals_per_weight = max(1, n_init // len(init_weights))

        for w in init_weights:
            if used >= budget or time.time() > deadline - 30:
                break

            # Start from a random point
            x0 = lo + (hi - lo) * rng.random(dim)
            f0 = problem(x0)
            used += 1

            # Short local search (hill climbing) for this weight
            best_x = x0.copy()
            best_f = f0.copy()
            best_scalar = np.max(w * best_f)

            for _ in range(evals_per_weight - 1):
                if used >= budget or time.time() > deadline - 30:
                    break

                # Generate perturbation
                step = 0.1 * (hi - lo)
                perturbation = step * rng.standard_normal(dim)
                x_new = np.clip(best_x + perturbation, lo, hi)
                f_new = problem(x_new)
                used += 1

                scalar_new = np.max(w * f_new)
                if scalar_new < best_scalar:
                    best_x = x_new
                    best_f = f_new
                    best_scalar = scalar_new

            # Add to archive
            archive_X, archive_F, _ = update_archive(archive_X, archive_F, best_x, best_f)

        # --- Phase 2: Simulated Annealing refinement on each archive point ---
        # Convert to arrays for easier handling
        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        if len(archive_X) == 0:
            # Fallback: random points
            for _ in range(min(budget - used, 10)):
                x = lo + (hi - lo) * rng.random(dim)
                f = problem(x)
                used += 1
                archive_X = np.vstack([archive_X, x.reshape(1, -1)]) if len(archive_X) > 0 else x.reshape(1, -1)
                archive_F = np.vstack([archive_F, f.reshape(1, -1)]) if len(archive_F) > 0 else f.reshape(1, -1)

        # SA refinement
        n_sa_iters = budget - used
        if n_sa_iters > 0 and len(archive_X) > 0:
            # Temperature schedule
            T0 = 0.1
            T_end = 0.001
            cooling_rate = (T_end / T0) ** (1.0 / max(1, n_sa_iters))

            # For each point in archive, do SA with random weights
            points_per_sa = max(1, n_sa_iters // len(archive_X))

            for i in range(len(archive_X)):
                if used >= budget or time.time() > deadline - 10:
                    break

                x = archive_X[i].copy()
                f = archive_F[i].copy()

                # Pick a random weight for this SA run
                w = rng.dirichlet(np.ones(n_obj))

                T = T0
                for sa_iter in range(points_per_sa):
                    if used >= budget or time.time() > deadline - 10:
                        break

                    # Generate perturbation
                    step = 0.05 * (hi - lo)
                    perturbation = step * rng.standard_normal(dim)
                    x_new = np.clip(x + perturbation, lo, hi)
                    f_new = problem(x_new)
                    used += 1

                    # Accept if better or with probability exp(-delta/T)
                    if dominates(f_new, f):
                        x = x_new
                        f = f_new
                    else:
                        # Compute scalar difference
                        scalar_old = np.max(w * f)
                        scalar_new = np.max(w * f_new)
                        delta = scalar_new - scalar_old
                        if delta < 0:
                            x = x_new
                            f = f_new
                        elif rng.random() < np.exp(-delta / max(T, 1e-10)):
                            x = x_new
                            f = f_new

                    T *= cooling_rate

                # Update archive with SA result
                archive_X, archive_F, _ = update_archive(archive_X, archive_F, x, f)

        # --- Phase 3: Final polish with very small steps ---
        remaining = budget - used
        if remaining > 0 and len(archive_X) > 0:
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                if used >= budget:
                    break

                # Pick a random archive point
                idx = rng.integers(len(archive_X))
                x = archive_X[idx].copy()
                f = archive_F[idx].copy()

                # Very small perturbation
                step = 0.01 * (hi - lo)
                perturbation = step * rng.standard_normal(dim)
                x_new = np.clip(x + perturbation, lo, hi)
                f_new = problem(x_new)
                used += 1

                archive_X, archive_F, _ = update_archive(archive_X, archive_F, x_new, f_new)

        # Convert to arrays if they're lists
        if isinstance(archive_X, list):
            archive_X = np.array(archive_X)
        if isinstance(archive_F, list):
            archive_F = np.array(archive_F)

        return archive_X

# EVOLVE-BLOCK-END
