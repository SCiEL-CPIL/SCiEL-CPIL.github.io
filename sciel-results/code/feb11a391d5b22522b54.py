"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for WFG7 using scalarization + L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        budget = self.budget
        n_obj = self.n_obj
        dim = self.dim
        ref = np.array([3.0, 5.0, 7.0])

        # Archive: list of (x, f) tuples
        archive_x = []
        archive_f = []
        used = 0

        def add_point(x, f):
            nonlocal used
            archive_x.append(x.copy())
            archive_f.append(f.copy())
            used += 1

        def get_nondominated_indices(F):
            """Return indices of non-dominated points in F (n x n_obj array)."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            # F[i] dominates F[j] if F[i] <= F[j] and F[i] != F[j]
            # Non-dominated: no other point dominates it
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        # Phase 1: Dense random sampling to build initial archive
        n_init = min(budget, 80)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        for x in X_init:
            f = problem(x)
            add_point(x, f)

        # Get initial non-dominated set
        F_arr = np.array(archive_f)
        nd_idx = get_nondominated_indices(F_arr)

        # Phase 2: Scalarization with L-BFGS-B
        # Generate weight vectors on 2D simplex for 3 objectives
        # w1 + w2 + w3 = 1, all >= 0
        n_weights = 25  # number of weight vectors
        # Create a grid of weight vectors
        weights = []
        # Use a stratified approach: corners + edges + interior
        # Corners
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        # Edge midpoints
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        # Interior points - use a triangular grid
        n_interior = n_weights - 6
        if n_interior > 0:
            # Generate points on simplex using Dirichlet-like sampling
            for _ in range(n_interior):
                w = rng.dirichlet([1.0, 1.0, 1.0])
                weights.append(w)

        weights = np.array(weights)
        # Normalize to ensure they sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)

        # For each weight, run L-BFGS-B from a good starting point
        # Starting point: pick from current non-dominated set the point that
        # is best aligned with the weight direction (lowest weighted sum)

        F_arr = np.array(archive_f)
        X_arr = np.array(archive_x)

        # For each weight, find the best starting point from the archive
        # and run L-BFGS-B
        for wi in range(len(weights)):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            w = weights[wi]

            # Find best starting point: minimize weighted sum in current archive
            weighted_sums = F_arr @ w
            start_idx = np.argmin(weighted_sums)
            x0 = X_arr[start_idx].copy()

            # Define the scalarized objective
            # Chebyshev scalarization: max_i (w_i * (f_i - z_i)) + epsilon * sum(f)
            # where z is an ideal point (min of each objective)
            # For simplicity, use weighted sum with a small Tchebycheff component

            def objective(x):
                f = problem(x)
                # Weighted sum
                ws = np.dot(w, f)
                return ws

            # Run L-BFGS-B
            try:
                bounds_list = list(zip(lo, hi))
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 30, 'maxfun': 30}
                )
            except Exception:
                continue

            # Add the result to archive
            x_best = result.x
            f_best = problem(x_best)
            add_point(x_best, f_best)

            # Also add some intermediate points if we have budget
            # L-BFGS-B doesn't easily give intermediate points, so we do
            # a few random perturbations around the result
            if used < budget:
                for _ in range(2):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    perturbation = rng.standard_normal(dim) * 0.05 * (hi - lo)
                    x_pert = x_best + perturbation
                    x_pert = np.clip(x_pert, lo, hi)
                    f_pert = problem(x_pert)
                    add_point(x_pert, f_pert)

        # Phase 3: Local refinement around non-dominated points
        # Get current non-dominated set
        F_arr = np.array(archive_f)
        X_arr = np.array(archive_x)
        nd_idx = get_nondominated_indices(F_arr)

        if len(nd_idx) > 0 and used < budget:
            # Sort non-dominated points by distance to reference point
            nd_F = F_arr[nd_idx]
            dists = np.sum((nd_F - ref) ** 2, axis=1)
            sorted_nd = nd_idx[np.argsort(dists)]

            # Refine the best non-dominated points
            n_refine = min(len(sorted_nd), max(1, (budget - used) // 8))
            for i in range(n_refine):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                base_idx = sorted_nd[i]
                base_x = X_arr[base_idx].copy()
                base_f = F_arr[base_idx].copy()

                # Local search: try small perturbations
                local_budget = min(8, budget - used)
                local_step = 0.03 * (hi - lo)

                best_x = base_x.copy()
                best_f = base_f.copy()

                for j in range(local_budget):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    direction = rng.standard_normal(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    step = local_step * (0.5 + 0.5 * rng.random())
                    candidate = best_x + step * direction
                    candidate = np.clip(candidate, lo, hi)

                    fc = problem(candidate)
                    add_point(candidate, fc)

                    # If candidate dominates best_f, update
                    if np.all(fc <= best_f) and np.any(fc < best_f):
                        best_x = candidate.copy()
                        best_f = fc.copy()

                # Update the archive with the best found
                # The best is already in the archive from add_point

        # Return all x points (the evaluator records all objective vectors)
        if len(archive_x) > 0:
            return np.array(archive_x)
        else:
            return np.array([lo + (hi - lo) * 0.5])

# EVOLVE-BLOCK-END
